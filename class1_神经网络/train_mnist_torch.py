"""
MNIST 训练 - PyTorch 版自动求导
与手写版完全相同的网络结构: 784 → 128(Sigmoid) → 10(Softmax)
输出 → output/torch/
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torchvision.datasets as datasets

# ============================================================
# 0. 创建输出目录 & 配置日志
# ============================================================
OUT_DIR = "output/torch"
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

# 划分
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=10000, random_state=42, stratify=y_all
)

logger.info(f"训练集: {X_train.shape[0]:>6,}")
logger.info(f"验证集: {X_val.shape[0]:>6,}")
logger.info(f"测试集: {X_test.shape[0]:>6,}")
logger.info("")

# ============================================================
# 2. PyTorch 模型定义
# ============================================================
class TwoLayerTorch(nn.Module):
    """与手写版完全相同的网络结构"""
    def __init__(self, n_input=784, n_hidden=128, n_output=10):
        super().__init__()
        self.hidden = nn.Linear(n_input, n_hidden)
        self.output = nn.Linear(n_hidden, n_output)
        # Xavier 初始化
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = torch.sigmoid(self.hidden(x))   # 与手写版一致: Sigmoid
        y = self.output(h)
        return y

# ============================================================
# 3. 训练
# ============================================================
logger.info("=" * 55)
logger.info("PyTorch 版")
logger.info("=" * 55)

device = torch.device("cpu")
model = TwoLayerTorch().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.5)
criterion = nn.CrossEntropyLoss()
batch_size = 64
n_iter = 100

# 转张量
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)


n_samples = X_train.shape[0]
n_batches = n_samples // batch_size

loss_history = []
val_acc_history = []

logger.info(f"网络: 784 → 128(Sigmoid) → 10")
logger.info(f"batch_size={batch_size}, {n_batches} batch/epoch\n")

for epoch in range(n_iter):
    # 打乱
    perm = torch.randperm(n_samples)
    X_shuffled = X_train_t[perm]
    y_shuffled = y_train_t[perm]

    epoch_loss = 0
    for b in range(n_batches):
        start = b * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * (end - start) / n_samples

    loss_history.append(epoch_loss)

    # 验证
    with torch.no_grad():
        val_pred = model(X_val_t)
        val_acc = (val_pred.argmax(1) == y_val_t).float().mean().item()
        val_acc_history.append(val_acc)

    if (epoch + 1) % 20 == 0:
        logger.info(f"Epoch {epoch+1:3d}/{n_iter} | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

# ============================================================
# 4. 评估
# ============================================================
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

with torch.no_grad():
    train_pred = model(X_train_t)
    train_acc = (train_pred.argmax(1) == y_train_t).float().mean().item()
    test_pred = model(X_test_t)
    test_acc = (test_pred.argmax(1) == y_test_t).float().mean().item()

logger.info("PyTorch 版结果:")
logger.info(f"  训练集: {train_acc*100:.2f}%")
logger.info(f"  验证集: {val_acc_history[-1]*100:.2f}%")
logger.info(f"  测试集: {test_acc*100:.2f}%")

# ============================================================
# 5. 保存模型
# ============================================================
torch.save(model.state_dict(), os.path.join(OUT_DIR, "mnist_torch_model.pth"))
logger.info(f"已保存: {OUT_DIR}/mnist_torch_model.pth")

np.savez(os.path.join(OUT_DIR, "mnist_torch_scaler.npz"),
         scaler_min=scaler.min_,
         scaler_scale=scaler.scale_,
         scaler_data_min=scaler.data_min_,
         scaler_data_max=scaler.data_max_)
logger.info(f"已保存: {OUT_DIR}/mnist_torch_scaler.npz")

np.savez(os.path.join(OUT_DIR, "mnist_torch_curve.npz"),
         loss=loss_history,
         val_acc=val_acc_history)
logger.info(f"已保存: {OUT_DIR}/mnist_torch_curve.npz")

# ============================================================
# 6. 保存训练曲线图
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# 损失曲线
ax1.plot(loss_history, "b-", linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax1.grid(True, alpha=0.3)

# 验证准确率曲线
ax2.plot(val_acc_history, "r-", linewidth=1.5)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Accuracy")
ax2.set_title("Validation Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curve.png"), dpi=150)
plt.close()
logger.info(f"已保存: {OUT_DIR}/training_curve.png")
