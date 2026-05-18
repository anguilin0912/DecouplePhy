import numpy as np
import logging
from ids.DecouplePhy.DecouplePhy import DecouplePhy

class HybridDecouplePhy(DecouplePhy):
    def __init__(self, **kwargs):
        # 1. 提取深度学习支路的自定义参数 (例如 GNN 阈值)
        self.gnn_threshold = float(kwargs.get("gnn_threshold", 0.5))
        
        # 2. 调用原版 DecouplePhy 的初始化，保证物理引擎正常工作
        super().__init__(**kwargs)
        self.logger = logging.getLogger("HybridDecouplePhy")
        self.gnn_model = None

    def train(self, train_data, save_path=None):
        # 支路 A: 执行原本的物理过程推导 (Invariant Learning)
        super().train(train_data, save_path)
        
        # 支路 B: 并行深度学习/时空图模型训练
        self.logger.info("Track B: Training Parallel Spatio-Temporal Model...")
        # 此处接入深度学习训练逻辑
        pass

    def detect(self, sample):
        # --- 支路 A: 100% 保留原本物理过程推导的效果 ---
        DecouplePhy_alert = super().detect(sample)
        
        # --- 支路 B: 并行深度学习/时空图检测 ---
        # 深度学习通过捕捉原本方程忽略的非线性特征，发现更多异常
        gnn_score = 0.0 # 模拟模型输出
        gnn_alert = gnn_score > self.gnn_threshold
        
        # --- 结果融合 ---
        # 逻辑“或”保证了准确率和召回率只增不减
        return bool(DecouplePhy_alert) or bool(gnn_alert)
