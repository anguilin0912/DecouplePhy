import copy
import json
import multiprocessing
import time
from itertools import chain, combinations, islice, product
from math import comb

import numpy as np
from scipy.optimize import curve_fit

import ipal_iids.settings as settings
from ids.DecouplePhy.equations import equations, get_equation
from ids.ids import MetaIDS

_DATAPOINTS = {}
_EPS = 0.0000000001

# --- [多进程物理方程寻优] ---
def _system_identification(args):
    global _DATAPOINTS
    target, equation, combination, std_factor = args

    if target in combination: return None, None
    if equation.name == "Product" and len(combination) == 0: return None, None

    combination = (target,) + combination
    x_data = np.array([_DATAPOINTS[s][:-1] for s in combination])
    y_data = np.array(_DATAPOINTS[target][1:])

    try:
        SPLIT = int(0.8 * len(y_data))
        x_train = np.array([x[:SPLIT] for x in x_data])
        y_train = np.array(y_data[:SPLIT])
        potp, _ = curve_fit(equation.calc, x_train, y_train, p0=equation.default_parameters(combination))
    except RuntimeError:
        return None, None

    errors = equation.calc(x_data, *potp) - y_data

    drift = np.mean(abs(errors)) + std_factor * np.std(errors)
    cusum = 0
    threshold = 0
    for err in errors:
        cusum = max(cusum + abs(err) - drift, 0)
        threshold = max(threshold, cusum)

    threshold = max(threshold, 1e-6)

    return target, {
        "error": np.mean(errors ** 2), "equation": equation, "combination": combination,
        "parameters": potp, "drift": drift, "threshold": threshold
    }

class DecouplePhy(MetaIDS):
    _name = "DecouplePhy"
    _description = "Ablation: Track 1 Physics Only"
    _requires = ["train.state", "live.state"]

    _DecouplePhy_default_settings = {
        "ignore": [], "max_formel_length": 1, "threshold_factor": 1.1, "cusum_factor": 2, "cpus": 4,
        "std_factor": 1.0
    }
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._DecouplePhy_default_settings)

        self.CI = {}
        self.last_value = {}
        self.cusum = {}
        self.sensor_order = []
        
        self.stats = {
            "phy_only": 0, "total_steps": 0,
            "track1_train_time": 0.0, "total_infer_time": 0.0
        }

    def __del__(self):
        if self.stats.get("total_steps", 0) > 0:
            print("\n" + "🚀 " * 15)
            print(f"[{self._name}] 消融实验：仅物理轨道 (Track 1) 评估报告")
            print("=" * 45)
            print(f"1. 物理轨道 训练耗时 : {self.stats.get('track1_train_time', 0):.4f} 秒")
            
            total_infer = self.stats.get('total_infer_time', 0)
            steps = self.stats.get('total_steps', 0)
            avg_infer = (total_infer / steps * 1000) if steps > 0 else 0
            print(f"2. 实时推理 总计耗时    : {total_infer:.4f} 秒")
            print(f"   └─ 平均单步推理      : {avg_infer:.4f} 毫秒/步")
            print("=" * 45 + "\n")

    def _all_combinations(self, sensor_names):
        f_len = range(0, self.settings["max_formel_length"] + 1)
        tests = chain(*[combinations(sensor_names, r) for r in f_len])
        count = sum([comb(len(sensor_names), r) for r in f_len])
        formulas = product(sensor_names, equations, tests)
        return int(len(sensor_names) * len(equations) * count), formulas

    def _batch(self, it, n, std_factor):
        assert n > 1
        while batch := tuple(islice(iter(it), n)):
            yield [(t, eq, c, std_factor) for t, eq, c in batch]

    def train(self, ipal=None, state=None):
        global _DATAPOINTS

        settings.logger.info(f"[{self._name}] Loading training data...")
        with self._open_file(state) as f:
            states = [json.loads(line)["state"] for line in f.readlines()]

        sensor_names = [s for s in states[0] if s not in self.settings["ignore"]]
        _DATAPOINTS = {sensor: [state[sensor] for state in states] for sensor in sensor_names}
        self.sensor_order = sensor_names.copy()
        del states

        if len(sensor_names) - 1 < self.settings["max_formel_length"]:
            self.settings["max_formel_length"] = len(sensor_names) - 1

        count, formulas = self._all_combinations(sensor_names)
        std_factor = float(self.settings.get("std_factor", 1.0))

        settings.logger.info(f"[{self._name}] Track 1: Mining physics with std_factor={std_factor}...")
        t1_start = time.time()

        with multiprocessing.get_context("fork").Pool(self.settings["cpus"]) as p:
            for batch in self._batch(formulas, self.settings["cpus"] * 100, std_factor):
                results = p.map(_system_identification, batch)
                for target, res in results:
                    if res is None: continue
                    if target not in self.CI or res["error"] < self.CI[target]["error"]:
                        self.CI[target] = res

        self.stats["track1_train_time"] = time.time() - t1_start
        self.save_trained_model(incomplete=False)

    def new_state_msg(self, msg):
        infer_start = time.time()
        phy_alert = False
        state = msg["state"]

        if len(self.cusum) == 0:
            self.cusum = {s: 0 for s in self.CI}
            self.last_value = {s: state[s] for s in state}
            self.stats["total_infer_time"] += (time.time() - infer_start)
            return False, [self.cusum[s] for s in self.CI]

        for sensor in self.sensor_order:
            if sensor in self.CI:
                ci = self.CI[sensor]
                X = [self.last_value.get(s, 0.0) for s in ci["combination"]]
                estimate = ci["equation"].calc(X, *ci["parameters"])
                diff = state.get(sensor, 0.0) - estimate
                
                self.cusum[sensor] = max(self.cusum.get(sensor, 0) + abs(diff) - ci["drift"] - _EPS, 0)
                self.cusum[sensor] = min(self.cusum[sensor],
                                         ci["threshold"] * self.settings["threshold_factor"] + ci["drift"] * self.settings["cusum_factor"] + _EPS)
                if self.cusum[sensor] > ci["threshold"] * self.settings["threshold_factor"]:
                    phy_alert = True
            
        self.last_value = state
        self.stats["total_steps"] += 1
        if phy_alert: self.stats["phy_only"] += 1
        self.stats["total_infer_time"] += (time.time() - infer_start)

        # 仅返回物理报警
        return phy_alert, [self.cusum.get(s, 0) for s in self.CI]

    def save_trained_model(self, incomplete=False):
        if self.settings["model-file"] is None: return False
        CI = copy.deepcopy(self.CI)
        for sensor in CI:
            CI[sensor]["equation"] = CI[sensor]["equation"].name
            CI[sensor]["parameters"] = list(CI[sensor]["parameters"])

        model = {
            "_name": self._name, "settings": self.settings, "incomplete": incomplete, "CI": CI,
            "sensor_order": self.sensor_order
        }
        with self._open_file(self._resolve_model_file_path(), mode="wt") as f:
            f.write(json.dumps(model, indent=4) + "\n")
        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None: return False
        try:
            with self._open_file(self._resolve_model_file_path(), mode="rt") as f:
                model = json.load(f)
        except FileNotFoundError:
            return False

        self.CI = model.get("CI", {})
        for sensor in self.CI: self.CI[sensor]["equation"] = get_equation(self.CI[sensor]["equation"])
        self.sensor_order = model.get("sensor_order", [])
        return True

    def visualize_model(self):
        return None, None