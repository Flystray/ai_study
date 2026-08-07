"""
MNIST 训练 - 手写梯度下降版
使用 neural_network.py 的 TwoLayerNN（Sigmoid + Mini-batch）
输出 → output/hand/
"""
import os
import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torchvision.datasets as datasets
from neural_network import TwoLayerNN

# ============================================================
# 0. 创建输出目录 & 配置日志
# ============================================================
OUT_DIR = "output/hand"
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, "training.log"), mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class LoggerWriter:
    """把 print() 输出重定向到 logger，保证日志格式统一"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger, logging.INFO)

# ============================================================
# 1. 下载 & 预处理
# ============================================================
logger.info("=" * 55)
logger.info("加载 MNIST...")
mnist = datasets.MNIST(root="./mnist_data", train=True, download=False)
X_all = mnist.data.numpy().reshape(-1, 784).astype(np.float64)
y_all = mnist.targets.numpy()

mnist_test = datasets.MNIST(root="./mnist_data", train=False, download=False)
X_test = mnist_test.data.numpy().reshape(-1, 784).astype(np.float64)
y_test = mnist_test.targets.numpy()

# 归一化
scaler = MinMaxScaler()
X_all = scaler.fit_transform(X_all)
X_test = scaler.transform(X_test)

# 划分 50000 训练, 10000 验证
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=10000, random_state=42, stratify=y_all
)

logger.info(f"训练集: {X_train.shape[0]:>6,}  维度: {X_train.shape[1]}")
logger.info(f"验证集: {X_val.shape[0]:>6,}")
logger.info(f"测试集: {X_test.shape[0]:>6,}")
logger.info("")

# ============================================================
# 2. 手写版训练
# ============================================================
logger.info("=" * 55)
logger.info("手写梯度下降版")
logger.info("=" * 55)

model_hand = TwoLayerNN(
    n_hidden=128,
    learning_rate=0.5,
    n_iter=100,           # 100 轮 (约 3-5 分钟)
    batch_size=64,
    verbose=True
)

logger.info(f"网络: {X_train.shape[1]} → {model_hand.n_hidden} → 10")
logger.info(f"激活: Sigmoid")
logger.info("")

val_acc_hand = model_hand.fit(X_train, y_train, X_val, y_val)

train_acc_hand = model_hand.score(X_train, y_train)
test_acc_hand = model_hand.score(X_test, y_test)

logger.info("手写版结果:")
logger.info(f"  训练集: {train_acc_hand*100:.2f}%")
logger.info(f"  验证集: {val_acc_hand[-1]*100:.2f}%")
logger.info(f"  测试集: {test_acc_hand*100:.2f}%")

# ============================================================
# 3. 保存模型
# ============================================================
np.savez(os.path.join(OUT_DIR, "mnist_hand_model.npz"),
         W1=model_hand.W1, b1=model_hand.b1,
         W2=model_hand.W2, b2=model_hand.b2,
         scaler_min=scaler.min_,
         scaler_scale=scaler.scale_,
         scaler_data_min=scaler.data_min_,
         scaler_data_max=scaler.data_max_)
logger.info(f"已保存: {OUT_DIR}/mnist_hand_model.npz")

np.savez(os.path.join(OUT_DIR, "mnist_hand_curve.npz"),
         loss=model_hand.loss_history,
         val_acc=val_acc_hand)
logger.info(f"已保存: {OUT_DIR}/mnist_hand_curve.npz")

# ============================================================
# 4. 保存训练曲线图
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(model_hand.loss_history, "b-", linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

# 验证准确率曲线
ax2.plot(val_acc_hand, "r-", linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
plt.close()
logger.info(f"已保存: {OUT_DIR}/training_curve.png")