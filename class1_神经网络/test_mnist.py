"""
加载已训练的 MNIST 模型，跑测试集
用法:
  python test_mnist.py hand     # 测试手写版
  python test_mnist.py torch    # 测试 PyTorch 版
  python test_mnist.py both     # 测试两个版本并对比
输出 → output/
"""
import os
import logging
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torchvision.datasets as datasets

# ============================================================
# 0. 创建输出目录 & 配置日志
# ============================================================
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "test_result.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ============================================================
# 1. 加载测试集
# ============================================================
logger.info("加载 MNIST 测试集...")
mnist_test = datasets.MNIST(root="./mnist_data", train=False, download=False)
X_test = mnist_test.data.numpy().reshape(-1, 784).astype(np.float64)
y_test = mnist_test.targets.numpy()


def _build_scaler(data_min, data_max):
    """从保存的数组重建 MinMaxScaler"""
    s = MinMaxScaler()
    s.fit(np.zeros((1, data_min.shape[0])))  # 占位 fit
    s.data_min_ = data_min
    s.data_max_ = data_max
    s.min_ = data_min
    s.scale_ = 1.0 / (data_max - data_min + 1e-10)
    s.feature_range = (0, 1)
    return s


MODEL_DIRS = {
    "hand": os.path.join(OUT_DIR, "hand"),
    "torch": os.path.join(OUT_DIR, "torch"),
}


def load_hand_model():
    """加载手写版模型"""
    data = np.load(os.path.join(MODEL_DIRS["hand"], "mnist_hand_model.npz"))
    scaler = _build_scaler(data["scaler_data_min"], data["scaler_data_max"])
    X_scaled = scaler.transform(X_test)
    X_scaled = X_scaled.T  # (784, 10000)

    # 手写前向传播
    A1 = 1 / (1 + np.exp(-(data["W1"] @ X_scaled + data["b1"])))
    A2 = np.exp(data["W2"] @ A1 + data["b2"])
    A2 = A2 / np.sum(A2, axis=0, keepdims=True)
    y_pred = np.argmax(A2, axis=0)
    acc = np.mean(y_pred == y_test)
    return acc


def load_torch_model():
    """加载 PyTorch 版模型"""
    class TwoLayerTorch(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(784, 128)
            self.output = nn.Linear(128, 10)
        def forward(self, x):
            h = torch.sigmoid(self.hidden(x))
            return self.output(h)

    model = TwoLayerTorch()
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIRS["torch"], "mnist_torch_model.pth"),
        map_location="cpu"
    ))
    model.eval()

    # 加载标量器
    scaler_data = np.load(os.path.join(MODEL_DIRS["torch"], "mnist_torch_scaler.npz"))
    scaler = _build_scaler(scaler_data["scaler_data_min"], scaler_data["scaler_data_max"])
    X_scaled = scaler.transform(X_test)

    with torch.no_grad():
        pred = model(torch.FloatTensor(X_scaled))
        y_pred = pred.argmax(1).numpy()
    acc = np.mean(y_pred == y_test)
    return acc


# ============================================================
# 2. 执行
# ============================================================
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
acc_results = {}

if mode in ("hand", "both"):
    logger.info("测试手写版模型...")
    acc_hand = load_hand_model()
    acc_results["手写版"] = acc_hand
    logger.info(f"  测试集准确率: {acc_hand*100:.2f}%")

if mode in ("torch", "both"):
    logger.info("测试 PyTorch 版模型...")
    acc_torch = load_torch_model()
    acc_results["PyTorch版"] = acc_torch
    logger.info(f"  测试集准确率: {acc_torch*100:.2f}%")

# ============================================================
# 3. 保存对比图
# ============================================================
if len(acc_results) >= 1:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(acc_results.keys(), [v * 100 for v in acc_results.values()],
                  color=["#2196F3", "#FF5722"], width=0.4, edgecolor="white")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MNIST Test Accuracy Comparison")
    ax.set_ylim(90, 100)
    ax.grid(axis="y", alpha=0.3)

    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                f"{height:.2f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "test_result.png"), dpi=150)
    plt.close()
    logger.info(f"已保存: {OUT_DIR}/test_result.png")
