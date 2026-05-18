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
# 目录已修改为你自己的专属架构名称
from ids.DecouplePhy.equations import equations, get_equation
from ids.ids import MetaIDS

# --- [深度学习依赖与设备自适应] ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    HAS_TORCH = True
    # 自动选择设备：有GPU用GPU，没有用CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

_DATAPOINTS = {}
_EPS = 0.0000000001

# --- [时空 Transformer 自编码器] ---
if HAS_TORCH:
    class STTransformerAE(nn.Module):
        def __init__(self, num_sensors, d_model=64):
            super(STTransformerAE, self).__init__()
            self.input_proj = nn.Linear(num_sensors, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=4, dim_feedforward=128,
                batch_first=True, dropout=0.1
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.output_proj = nn.Linear(d_model, num_sensors)

        def forward(self, x):
            x_proj = self.input_proj(x)
            out = self.transformer(x_proj)
            return self.output_proj(out)


# --- [多进程物理方程寻优] ---
def _system_identification(args):
    global _DATAPOINTS
    # 动态接收外部传入的 std_factor (物理宽容度)
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

    # 【核心逻辑：基于动态 std_factor 构建自适应防线】
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
    # 核心架构冠名权
    _name = "DecouplePhy"
    _description = "DecouplePhy: Adaptive Dual-Track Architecture for ICS"
    _requires = ["train.state", "live.state"]

    # 【核心注册：在系统默认配置中暴露三大灵魂参数】
    _DecouplePhy_default_settings = {
        "ignore": [], "max_formel_length": 1, "threshold_factor": 1.1, "cusum_factor": 2, "cpus": 4,
        "std_factor": 1,  # 物理防线宽容度
        "dl_multiplier": 1,  # AI拓扑敏锐度
        "debounce": 5,  # 时域连续防抖窗口
        "epochs": 33  # AI 训练轮数
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
        self.window_size = 12
        self.state_buffer = deque(maxlen=self.window_size)
        self.sensor_max_errors = []
        self.anomaly_counter = 0

        # 统计数据：物理(phy)和深度学习(dl)
        self.stats = {
            "phy_only": 0, "dl_only": 0, "both": 0, "total_steps": 0,
            "track1_train_time": 0.0,
            "track2_train_time": 0.0,
            "total_infer_time": 0.0
        }

    # 析构函数：在模型运行彻底结束后，自动打印实验时间评估报告
    def __del__(self):
        if self.stats.get("total_steps", 0) > 0 or self.stats.get("track1_train_time", 0) > 0:
            print("\n" + "🚀 " * 15)
            print(f"[{self._name}] 运行耗时及效能评估报告")
            print("=" * 45)
            print(f"1. 物理轨道 (Track 1) 训练耗时 : {self.stats.get('track1_train_time', 0):.4f} 秒")
            print(f"2. 深度残差 (Track 2) 训练耗时 : {self.stats.get('track2_train_time', 0):.4f} 秒")

            total_infer = self.stats.get('total_infer_time', 0)
            steps = self.stats.get('total_steps', 0)
            avg_infer = (total_infer / steps * 1000) if steps > 0 else 0

            print(f"3. 实时推理 (Live) 总计耗时    : {total_infer:.4f} 秒")
            print(f"   └─ 平均单步推理极速         : {avg_infer:.4f} 毫秒/步")
            print("=" * 45 + "\n")

    def _all_combinations(self, sensor_names):
        f_len = range(0, self.settings["max_formel_length"] + 1)
        tests = chain(*[combinations(sensor_names, r) for r in f_len])
        count = sum([comb(len(sensor_names), r) for r in f_len])
        formulas = product(sensor_names, equations, tests)
        return int(len(sensor_names) * len(equations) * count), formulas

    # 批处理迭代器，强制将 std_factor 打包发给多进程 worker
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
        data_matrix = np.array([_DATAPOINTS[s] for s in self.sensor_order]).T
        del states

        if len(sensor_names) - 1 < self.settings["max_formel_length"]:
            self.settings["max_formel_length"] = len(sensor_names) - 1

        count, formulas = self._all_combinations(sensor_names)

        # 从 JSON 中提取动态超参数
        std_factor = float(self.settings.get("std_factor", 1.0))

        # --- 计时器 1：物理防线 (Track 1) 训练 ---
        settings.logger.info(f"[{self._name}] Track 1: Mining physics with std_factor={std_factor}...")
        t1_start = time.time()

        with multiprocessing.get_context("fork").Pool(self.settings["cpus"]) as p:
            for batch in self._batch(formulas, self.settings["cpus"] * 100, std_factor):
                results = p.map(_system_identification, batch)
                for target, res in results:
                    if res is None: continue
                    if target not in self.CI or res["error"] < self.CI[target]["error"]:
                        self.CI[target] = res

        t1_end = time.time()
        self.stats["track1_train_time"] = t1_end - t1_start

        # ==============================================================
        # 核心修复 1：提取纯净残差子空间 (Residual Subspace Extraction)
        # ==============================================================
        if HAS_TORCH:
            settings.logger.info(f"[{self._name}] Track 2: Extracting Residual Subspace...")
            
            # 初始化残差矩阵 (大小与原始数据相同)
            residual_matrix = np.zeros_like(data_matrix)
            last_val = {s: data_matrix[0, idx] for idx, s in enumerate(self.sensor_order)}

            # 遍历训练集，计算每一步的物理残差 R_t = X_t - \hat{X}_t
            for t_idx in range(1, len(data_matrix)):
                current_val = {s: data_matrix[t_idx, idx] for idx, s in enumerate(self.sensor_order)}
                for s_idx, sensor in enumerate(self.sensor_order):
                    if sensor in self.CI:
                        ci = self.CI[sensor]
                        X_input = [last_val[s] for s in ci["combination"]]
                        estimate = ci["equation"].calc(X_input, *ci["parameters"])
                        # 解耦：真实值减去物理预测值
                        residual_matrix[t_idx, s_idx] = current_val[sensor] - estimate
                    else:
                        # 若无物理方程，使用简单一阶差分作为残差
                        residual_matrix[t_idx, s_idx] = current_val[sensor] - last_val.get(sensor, current_val[sensor])
                last_val = current_val

            settings.logger.info(f"[{self._name}] Track 2: Training ST-Transformer on {DEVICE}...")
            t2_start = time.time()

            # 注意：现在是用 residual_matrix (残差矩阵) 来计算 min/max 并进行归一化！
            mins = np.min(residual_matrix, axis=0)
            maxs = np.max(residual_matrix, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            self.sensor_mins = mins.tolist()
            self.sensor_ranges = ranges.tolist()
            scaled_matrix = (residual_matrix - mins) / ranges

            X = []
            for i in range(len(scaled_matrix) - self.window_size + 1):
                X.append(scaled_matrix[i: i + self.window_size])

            # 数据搬运到对应设备
            X_tensor = torch.FloatTensor(np.array(X)).to(DEVICE)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            self.dl_model = STTransformerAE(num_sensors=len(self.sensor_order)).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.dl_model.parameters(), lr=0.003)

            self.dl_model.train()

            # 动态读取 epochs 配置
            train_epochs = int(self.settings.get("epochs", 25))
            for epoch in range(train_epochs):
                for batch_x, _ in dataloader:
                    optimizer.zero_grad()
                    loss = criterion(self.dl_model(batch_x), batch_x)
                    loss.backward()
                    optimizer.step()

            self.dl_model.eval()
            all_errors = []
            with torch.no_grad():
                # 分批次进行推理，防止超大 Tensor 撑爆算子
                eval_batch_size = 512
                for i in range(0, len(X_tensor), eval_batch_size):
                    batch_x = X_tensor[i: i + eval_batch_size]
                    preds = self.dl_model(batch_x)
                    batch_err = torch.mean(torch.abs(preds - batch_x), dim=1).cpu().numpy()
                    all_errors.append(batch_err)

                # 将所有批次的误差拼接起来计算最大阈值
                errors = np.concatenate(all_errors, axis=0)

                # 从 JSON 中获取 AI 敏锐度动态超参数
                dl_multi = float(self.settings.get("dl_multiplier", 1.4))
                self.sensor_max_errors = (np.max(errors, axis=0) * dl_multi).tolist()

            t2_end = time.time()
            self.stats["track2_train_time"] = t2_end - t2_start

        self.save_trained_model(incomplete=False)

    def new_state_msg(self, msg):
        # --- 计时器 3：在线实时推理耗时 ---
        infer_start = time.time()

        phy_alert = False
        gnn_alert = False
        state = msg["state"]

        if len(self.cusum) == 0:
            self.cusum = {s: 0 for s in self.CI}
            self.last_value = {s: state[s] for s in state}

            infer_end = time.time()
            self.stats["total_infer_time"] += (infer_end - infer_start)
            return False, [self.cusum[s] for s in self.CI]

        # ==============================================================
        # 核心修复 2：物理轨道 (Track 1) 与残差实时提取
        # ==============================================================
        current_residuals = [] # 专门用于收集当前时刻的残差向量 R_t
        
        for sensor in self.sensor_order:
            if sensor in self.CI:
                ci = self.CI[sensor]
                X = [self.last_value.get(s, 0.0) for s in ci["combination"]]
                estimate = ci["equation"].calc(X, *ci["parameters"])
                diff = state.get(sensor, 0.0) - estimate
                
                # CUSUM 物理防线判定
                self.cusum[sensor] = max(self.cusum.get(sensor, 0) + abs(diff) - ci["drift"] - _EPS, 0)
                self.cusum[sensor] = min(self.cusum[sensor],
                                         ci["threshold"] * self.settings["threshold_factor"] + ci["drift"] * self.settings["cusum_factor"] + _EPS)
                if self.cusum[sensor] > ci["threshold"] * self.settings["threshold_factor"]:
                    phy_alert = True
                
                # 收集残差
                current_residuals.append(diff)
            else:
                # 若不在物理轨道监控中，使用一阶差分作为残差
                diff = state.get(sensor, 0.0) - self.last_value.get(sensor, 0.0)
                current_residuals.append(diff)

        self.last_value = state

        # ==============================================================
        # 核心修复 3：AI 轨道 (Track 2) 接收残差空间数据
        # ==============================================================
        if HAS_TORCH and self.dl_model is not None and len(self.sensor_mins) > 0:
            # 使用提取出的 current_residuals 进行归一化
            scaled_res = (np.array(current_residuals) - np.array(self.sensor_mins)) / np.array(self.sensor_ranges)
            self.state_buffer.append(scaled_res.tolist())

            if len(self.state_buffer) == self.window_size:
                self.dl_model.eval()
                with torch.no_grad():
                    # 推理时将当前残差窗口搬运到模型所在设备
                    t_window = torch.FloatTensor([list(self.state_buffer)]).to(DEVICE)
                    # 结果拉回 CPU 处理
                    pred_window = self.dl_model(t_window)
                    current_errors = torch.mean(torch.abs(pred_window - t_window), dim=1).cpu().numpy()[0]

                    violating_sensors = sum(
                        [1 for i, err in enumerate(current_errors) if err > self.sensor_max_errors[i]])
                    if violating_sensors >= 2:
                        self.anomaly_counter += 1
                    else:
                        self.anomaly_counter = 0

                    # 从 JSON 中动态获取防抖窗口大小
                    debounce_limit = int(self.settings.get("debounce", 2))
                    if self.anomaly_counter >= debounce_limit:
                        gnn_alert = True

        self.stats["total_steps"] += 1
        if phy_alert and not gnn_alert:
            self.stats["phy_only"] += 1
        elif gnn_alert and not phy_alert:
            self.stats["dl_only"] += 1
        elif phy_alert and gnn_alert:
            self.stats["both"] += 1

        infer_end = time.time()
        self.stats["total_infer_time"] += (infer_end - infer_start)

        return phy_alert or gnn_alert, [self.cusum.get(s, 0) for s in self.CI]

    def save_trained_model(self, incomplete=False):
        if self.settings["model-file"] is None: return False
        CI = copy.deepcopy(self.CI)
        for sensor in CI:
            CI[sensor]["equation"] = CI[sensor]["equation"].name
            CI[sensor]["parameters"] = list(CI[sensor]["parameters"])

        dl_weights = {}
        if HAS_TORCH and self.dl_model is not None:
            # 权重强制存为 CPU 格式，保障边缘侧兼容性
            for k, v in self.dl_model.state_dict().items():
                dl_weights[k] = v.cpu().tolist()

        model = {
            "_name": self._name, "settings": self.settings, "incomplete": incomplete, "CI": CI,
            "sensor_order": self.sensor_order, "window_size": self.window_size,
            "sensor_mins": self.sensor_mins, "sensor_ranges": self.sensor_ranges,
            "sensor_max_errors": self.sensor_max_errors,
            "dl_weights": dl_weights
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
            # 加载时立刻搬运到当前可用设备
            self.dl_model = STTransformerAE(num_sensors=len(self.sensor_order)).to(DEVICE)
            state_dict = {k: torch.FloatTensor(v).to(DEVICE) for k, v in dl_weights.items()}
            self.dl_model.load_state_dict(state_dict)
            self.dl_model.eval()
        return True

    def visualize_model(self):
        return None, None