import copy
import json
import time
from collections import deque

import numpy as np

import ipal_iids.settings as settings
from ids.ids import MetaIDS

# --- [深度学习依赖与设备自适应] ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    HAS_TORCH = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_TORCH = False
    DEVICE = "cpu"

_EPS = 0.0000000001

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


class DecouplePhy(MetaIDS):
    _name = "DecouplePhy"
    _description = "Ablation: Track 2 Deep Learning Only (Raw Data End-to-End)"
    _requires = ["train.state", "live.state"]

    _DecouplePhy_default_settings = {
        "ignore": [], 
        "dl_multiplier": 1.0,  
        "debounce": 5,  
        "epochs": 33  
    }
    _supports_preprocessor = False

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._DecouplePhy_default_settings)

        self.dl_model = None
        self.sensor_order = []
        self.sensor_mins = []
        self.sensor_ranges = []
        self.window_size = 12
        self.state_buffer = deque(maxlen=self.window_size)
        self.sensor_max_errors = []
        self.anomaly_counter = 0

        self.stats = {
            "dl_only": 0, "total_steps": 0,
            "track2_train_time": 0.0, "total_infer_time": 0.0
        }

    def train(self, ipal=None, state=None):
        settings.logger.info(f"[{self._name}] Loading training data for Ablation (Track 2 Only)...")
        with self._open_file(state) as f:
            states = [json.loads(line)["state"] for line in f.readlines()]

        sensor_names = [s for s in states[0] if s not in self.settings["ignore"]]
        _DATAPOINTS = {sensor: [state[sensor] for state in states] for sensor in sensor_names}
        self.sensor_order = sensor_names.copy()
        
        # 【修复核心】：直接使用原始数据矩阵，不计算差分！退化为传统自编码器
        raw_matrix = np.array([_DATAPOINTS[s] for s in self.sensor_order]).T
        del states

        if HAS_TORCH:
            settings.logger.info(f"[{self._name}] Track 2 Ablation: Normalizing Raw Data (No Physics)...")
            t2_start = time.time()

            # 基于原始数据计算 min 和 max
            mins = np.min(raw_matrix, axis=0)
            maxs = np.max(raw_matrix, axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            self.sensor_mins = mins.tolist()
            self.sensor_ranges = ranges.tolist()
            
            scaled_matrix = (raw_matrix - mins) / ranges

            X = []
            for i in range(len(scaled_matrix) - self.window_size + 1):
                X.append(scaled_matrix[i: i + self.window_size])

            settings.logger.info(f"[{self._name}] Track 2: Training ST-Transformer on {DEVICE}...")
            X_tensor = torch.FloatTensor(np.array(X)).to(DEVICE)
            dataset = TensorDataset(X_tensor, X_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

            self.dl_model = STTransformerAE(num_sensors=len(self.sensor_order)).to(DEVICE)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.dl_model.parameters(), lr=0.003)

            self.dl_model.train()
            train_epochs = int(self.settings.get("epochs", 33))
            for epoch in range(train_epochs):
                for batch_x, _ in dataloader:
                    optimizer.zero_grad()
                    loss = criterion(self.dl_model(batch_x), batch_x)
                    loss.backward()
                    optimizer.step()

            self.dl_model.eval()
            all_errors = []
            with torch.no_grad():
                eval_batch_size = 512
                for i in range(0, len(X_tensor), eval_batch_size):
                    batch_x = X_tensor[i: i + eval_batch_size]
                    preds = self.dl_model(batch_x)
                    batch_err = torch.mean(torch.abs(preds - batch_x), dim=1).cpu().numpy()
                    all_errors.append(batch_err)

                errors = np.concatenate(all_errors, axis=0)
                dl_multi = float(self.settings.get("dl_multiplier", 1.0))
                # 提取训练集重构误差极大值
                self.sensor_max_errors = (np.max(errors, axis=0) * dl_multi).tolist()

            self.stats["track2_train_time"] = time.time() - t2_start

        self.save_trained_model(incomplete=False)

    def new_state_msg(self, msg):
        infer_start = time.time()
        gnn_alert = False
        state = msg["state"]

        current_raw = [] 
        # 直接提取原始值进行评估
        for sensor in self.sensor_order:
            current_raw.append(state.get(sensor, 0.0))

        if HAS_TORCH and self.dl_model is not None and len(self.sensor_mins) > 0:
            # 使用原始数据的 min/max 进行归一化
            scaled_val = (np.array(current_raw) - np.array(self.sensor_mins)) / np.array(self.sensor_ranges)
            self.state_buffer.append(scaled_val.tolist())

            if len(self.state_buffer) == self.window_size:
                self.dl_model.eval()
                with torch.no_grad():
                    t_window = torch.FloatTensor([list(self.state_buffer)]).to(DEVICE)
                    pred_window = self.dl_model(t_window)
                    current_errors = torch.mean(torch.abs(pred_window - t_window), dim=1).cpu().numpy()[0]

                    # 【防爆零修复】：加入 _EPS，防止死区报警
                    thresholds = np.array(self.sensor_max_errors) + _EPS

                    violating_sensors = sum(
                        [1 for i, err in enumerate(current_errors) if err > thresholds[i]])
                    
                    if violating_sensors >= 2:
                        self.anomaly_counter += 1
                    else:
                        self.anomaly_counter = 0

                    debounce_limit = int(self.settings.get("debounce", 5))
                    if self.anomaly_counter >= debounce_limit:
                        gnn_alert = True

        self.stats["total_steps"] += 1
        if gnn_alert: self.stats["dl_only"] += 1
        self.stats["total_infer_time"] += (time.time() - infer_start)

        return gnn_alert, []

    def save_trained_model(self, incomplete=False):
        if self.settings["model-file"] is None: return False
        
        dl_weights = {}
        if HAS_TORCH and self.dl_model is not None:
            for k, v in self.dl_model.state_dict().items():
                dl_weights[k] = v.cpu().tolist()

        model = {
            "_name": self._name, "settings": self.settings, "incomplete": incomplete,
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

        self.sensor_order = model.get("sensor_order", [])
        self.window_size = model.get("window_size", 12)
        self.sensor_mins = model.get("sensor_mins", [])
        self.sensor_ranges = model.get("sensor_ranges", [])
        self.sensor_max_errors = model.get("sensor_max_errors", [])
        self.state_buffer = deque(maxlen=self.window_size)
        self.anomaly_counter = 0

        dl_weights = model.get("dl_weights", {})
        if HAS_TORCH and dl_weights and self.sensor_order:
            self.dl_model = STTransformerAE(num_sensors=len(self.sensor_order)).to(DEVICE)
            state_dict = {k: torch.FloatTensor(v).to(DEVICE) for k, v in dl_weights.items()}
            self.dl_model.load_state_dict(state_dict)
            self.dl_model.eval()
        return True

    def visualize_model(self):
        return None, None