import copy
import json
import multiprocessing
import time
from collections import deque
from itertools import chain, combinations, islice, product
from math import comb

import numpy as np
from scipy.optimize import curve_fit

import ipal_iids.settings as settings
from ids.DecouplePhy.equations import equations, get_equation
from ids.ids import MetaIDS

# --- [深度学习依赖] ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

_DATAPOINTS = {}  
_EPS = 0.0000000001

# --- [终极大招: 时空 Transformer 自编码器] ---
# 完美捕捉长周期序列下的全局传感器隐性关联
if HAS_TORCH:
    class STTransformerAE(nn.Module):
        def __init__(self, num_sensors, d_model=64):
            super(STTransformerAE, self).__init__()
            # 1. 投影层：将传感器原始维度映射到高维空间
            self.input_proj = nn.Linear(num_sensors, d_model)
            
            # 2. Transformer 编码器：通过 Multi-Head Attention 计算时空图拓扑
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=4, 
                dim_feedforward=128, 
                batch_first=True,
                dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # 3. 解码层：还原传感器状态
            self.output_proj = nn.Linear(d_model, num_sensors)

        def forward(self, x):
            # x shape: (Batch, Window_Size, Num_Sensors)
            x_proj = self.input_proj(x)
            out = self.transformer(x_proj)
            recons = self.output_proj(out)
            return recons


def _system_identification(args):
    global _DATAPOINTS
    target, equation, combination = args

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
    drift = np.mean(abs(errors)) + np.std(errors)
    cusum = 0
    threshold = 0
    for err in errors:
        cusum = max(cusum + abs(err) - drift, 0)
        threshold = max(threshold, cusum)

    return target, {
        "error": np.mean(errors**2), "equation": equation, "combination": combination,
        "parameters": potp, "drift": drift, "threshold": threshold
    }


class DecouplePhy(MetaIDS):
    _name = "DecouplePhy"
    _description = "DecouplePhy + ST-Transformer Long-Window Branch"
    _requires = ["train.state", "live.state"]
    _DecouplePhy_default_settings = {
        "ignore": [], "max_formel_length": 3, "threshold_factor": 1.1, "cusum_factor": 2, "cpus": 1,
    }
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._DecouplePhy_default_settings)

        self.CI = {}
        self.last_value = {}
        self.cusum = {}
        
        self.dl_model = None
        self.sensor_order = []
        self.sensor_mins = []
        self.sensor_ranges = []
        
        # [核心增强] 将视野扩大到 12 步，捕获缓慢漂移攻击
        self.window_size = 12
        self.state_buffer = deque(maxlen=self.window_size)
        
        self.sensor_max_errors = []
        self.anomaly_counter = 0
        
        # 统计战报板
        self.stats = {"track1_only": 0, "track2_only": 0, "both": 0, "total_steps": 0}

    def _all_combinations(self, sensor_names):
        f_len = range(0, self.settings["max_formel_length"] + 1)
        tests = chain(*[combinations(sensor_names, r) for r in f_len])
        count = sum([comb(len(sensor_names), r) for r in f_len])
        formulas = product(sensor_names, equations, tests)
        return int(len(sensor_names) * len(equations) * count), formulas

    def _batch(self, it, n):
        assert n > 1
        while batch := tuple(islice(iter(it), n)): yield batch

    def train(self, ipal=None, state=None):
        global _DATAPOINTS

        # === 支路 A: 物理方程训练 ===
        settings.logger.info("Loading training data")
        with self._open_file(state) as f:
            states = [json.loads(line)["state"] for line in f.readlines()]

        sensor_names = [s for s in states[0] if s not in self.settings["ignore"]]
        _DATAPOINTS = {sensor: [state[sensor] for state in states] for sensor in sensor_names}
        self.sensor_order = sensor_names.copy()
        
        data_matrix = np.array([_DATAPOINTS[s] for s in self.sensor_order]).T
        del states 

        if len(sensor_names) - 1 < self.settings["max_formel_length"]:
            self.settings["max_formel_length"] = len(sensor_names) - 1

        count, formulas = self._all_combinations(sensor_names)
        
        settings.logger.info("Track A: Parallel mining physics invariants...")
        with multiprocessing.get_context("fork").Pool(self.settings["cpus"]) as p:
            for batch in self._batch(formulas, self.settings["cpus"] * 100):
                results = p.map(_system_identification, batch)
                for target, res in results:
                    if res is None: continue
                    if target not in self.CI or res["error"] < self.CI[target]["error"]:
                        self.CI[target] = res

        # === 支路 B: ST-Transformer 训练 ===
        if HAS_TORCH:
            settings.logger.info(f"Track B: Training ST-Transformer with Window Size {self.window_size}...")
            
            mins = np.min(data_matrix, axis=0)
            maxs = np.max(data_matrix, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0  
            self.sensor_mins = mins.tolist()
            self.sensor_ranges = ranges.tolist()
            scaled_matrix = (data_matrix - mins) / ranges
            
            # 构造滑动窗口
            X = []
            for i in range(len(scaled_matrix) - self.window_size + 1):
                X.append(scaled_matrix[i : i + self.window_size])
                
            X_tensor = torch.FloatTensor(np.array(X))
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            input_dim = len(self.sensor_order)
            self.dl_model = STTransformerAE(num_sensors=input_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.dl_model.parameters(), lr=0.003)
            
            self.dl_model.train()
            epochs = 25  
            for epoch in range(epochs):
                for batch_x, _ in dataloader:
                    optimizer.zero_grad()
                    preds = self.dl_model(batch_x)
                    loss = criterion(preds, batch_x)
                    loss.backward()
                    optimizer.step()
            
            # 计算每条传感器的长周期重构误差阈值
            self.dl_model.eval()
            with torch.no_grad():
                all_preds = self.dl_model(X_tensor)
                # errors shape: (Batch, Window_Size, Sensors)
                # 我们取整个窗口内的平均绝对误差作为特征
                errors = torch.mean(torch.abs(all_preds - X_tensor), dim=1).numpy()
                
                # 保守阈值：最大正常误差的 1.25 倍
                self.sensor_max_errors = (np.max(errors, axis=0) * 1.25).tolist()
                
            settings.logger.info("Track B ST-Transformer trained successfully.")

        self.save_trained_model(incomplete=False)

    def new_state_msg(self, msg):
        DecouplePhy_alert = False
        gnn_alert = False
        state = msg["state"]

        if len(self.cusum) == 0:
            self.cusum = {s: 0 for s in self.CI}
            self.last_value = {s: state[s] for s in state}
            return False, [self.cusum[s] for s in self.CI]

        # === 支路 A: 物理 CUSUM ===
        for sensor, ci in self.CI.items():
            X = [self.last_value[s] for s in ci["combination"]]
            estimate = ci["equation"].calc(X, *ci["parameters"])
            diff = estimate - state[sensor]
            self.cusum[sensor] = max(self.cusum[sensor] + abs(diff) - ci["drift"] - _EPS, 0)
            self.cusum[sensor] = min(self.cusum[sensor], ci["threshold"] * self.settings["threshold_factor"] + ci["drift"] * self.settings["cusum_factor"] + _EPS)

            if self.cusum[sensor] > ci["threshold"] * self.settings["threshold_factor"]:
                DecouplePhy_alert = True
        
        self.last_value = state

        # === 支路 B: Transformer 时空检测 ===
        current_feat = [state.get(s, 0.0) for s in self.sensor_order]
        
        if HAS_TORCH and getattr(self, "dl_model", None) is not None and len(self.sensor_mins) > 0:
            scaled_feat = (np.array(current_feat) - np.array(self.sensor_mins)) / np.array(self.sensor_ranges)
            self.state_buffer.append(scaled_feat.tolist())
            
            if len(self.state_buffer) == self.window_size:
                self.dl_model.eval()
                with torch.no_grad():
                    t_window = torch.FloatTensor([list(self.state_buffer)])
                    pred_window = self.dl_model(t_window)
                    
                    # 计算当前窗口的时空预测偏差
                    current_errors = torch.mean(torch.abs(pred_window - t_window), dim=1).numpy()[0]
                    
                    violating_sensors = sum([1 for i, err in enumerate(current_errors) if err > self.sensor_max_errors[i]])
                    
                    # 只要有 >=2 个传感器脱离全局拓扑关系就累计
                    if violating_sensors >= 2:
                        self.anomaly_counter += 1
                    else:
                        self.anomaly_counter = 0
                        
                    # 连续异常验证
                    if self.anomaly_counter >= 2:
                        gnn_alert = True

        # === [战报统计] ===
        self.stats["total_steps"] += 1
        if DecouplePhy_alert and not gnn_alert: self.stats["DecouplePhy_only"] += 1
        elif gnn_alert and not DecouplePhy_alert: self.stats["dl_only"] += 1
        elif DecouplePhy_alert and gnn_alert: self.stats["both"] += 1

        if self.stats["total_steps"] % 1000 == 0:
            settings.logger.info(
                f"[Live Battle Report] DecouplePhy Only: {self.stats['DecouplePhy_only']} | "
                f"DL Only (Transformer): {self.stats['dl_only']} | Overlap: {self.stats['both']}"
            )

        final_alert = DecouplePhy_alert or gnn_alert
        return final_alert, [self.cusum[s] for s in self.CI]

    def save_trained_model(self, incomplete=False):
        if self.settings["model-file"] is None: return False
        CI = copy.deepcopy(self.CI)
        for sensor in CI:
            CI[sensor]["equation"] = CI[sensor]["equation"].name
            CI[sensor]["parameters"] = list(CI[sensor]["parameters"])
            
        dl_weights = {}
        if HAS_TORCH and getattr(self, "dl_model", None) is not None:
            for k, v in self.dl_model.state_dict().items(): dl_weights[k] = v.tolist()

        model = {
            "_name": self._name, "settings": self.settings, "incomplete": incomplete, "CI": CI,
            "sensor_order": getattr(self, "sensor_order", []), "window_size": getattr(self, "window_size", 12),
            "sensor_mins": getattr(self, "sensor_mins", []), "sensor_ranges": getattr(self, "sensor_ranges", []),
            "sensor_max_errors": getattr(self, "sensor_max_errors", []),
            "dl_weights": dl_weights
        }
        with self._open_file(self._resolve_model_file_path(), mode="wt") as f:
            f.write(json.dumps(model, indent=4) + "\n")
        return True

    def load_trained_model(self):
        if self.settings["model-file"] is None: return False
        try:
            with self._open_file(self._resolve_model_file_path(), mode="rt") as f: model = json.load(f)
        except FileNotFoundError: return False
        assert self._name == model["_name"]
        
        self.CI = model["CI"]
        for sensor in self.CI: self.CI[sensor]["equation"] = get_equation(self.CI[sensor]["equation"])
        
        self.sensor_order = model.get("sensor_order", [])
        self.window_size = model.get("window_size", 12)
        self.sensor_mins = model.get("sensor_mins", [])
        self.sensor_ranges = model.get("sensor_ranges", [])
        self.sensor_max_errors = model.get("sensor_max_errors", [])
        self.state_buffer = deque(maxlen=self.window_size)
        self.anomaly_counter = 0
        
        dl_weights = model.get("dl_weights", {})
        if HAS_TORCH and dl_weights and self.sensor_order:
            self.dl_model = STTransformerAE(num_sensors=len(self.sensor_order))
            state_dict = {}
            for k, v in dl_weights.items(): state_dict[k] = torch.FloatTensor(v)
            self.dl_model.load_state_dict(state_dict)
            self.dl_model.eval()
        else: self.dl_model = None
            
        return True

    def visualize_model(self): return None, None
