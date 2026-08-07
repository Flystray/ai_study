# -*- coding: utf-8 -*-
"""
公共工具模块：
- 数据集加载（MNIST / Fashion-MNIST / CIFAR-10）
- 网络定义（可配置 LeNet-5、全连接网络 MLP）
- 训练 / 评估函数
- 绘图与结果保存
所有实验统一依赖此模块，保证超参口径一致。
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")  # 无界面后端，保存图片
import matplotlib.pyplot as plt

# ---------------- 中文字体（Windows） ----------------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------- 全局路径 ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 运行设备（CPU 环境）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据集加载
# ============================================================
def load_dataset(name, batch_size=128, num_workers=0, normalize=True):
    """加载数据集，返回 (train_loader, test_loader, in_channels, height, width, num_classes, class_names)
    normalize=False 时输入保持 [0,1]（自编码器等需要与 sigmoid 输出同量纲）。"""
    if name in ("mnist", "fashion", "cifar10"):
        pass
    else:
        raise ValueError(name)

    if name in ("mnist", "fashion"):
        t = [T.ToTensor()]
        if normalize:
            t.append(T.Normalize((0.5,), (0.5,)))
        norm = T.Compose(t)
        cls = torchvision.datasets.MNIST if name == "mnist" else torchvision.datasets.FashionMNIST
        train = cls(root=DATA_DIR, train=True, download=True, transform=norm)
        test = cls(root=DATA_DIR, train=False, download=True, transform=norm)
        classes = [str(i) for i in range(10)]
        in_ch, h, w = 1, 28, 28
    else:  # cifar10
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_tf)
        test = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_tf)
        classes = train.classes
        in_ch, h, w = 3, 32, 32

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, in_ch, h, w, 10, classes


# ============================================================
# 网络定义
# ============================================================
class LeNet5(nn.Module):
    """
    可配置的 LeNet-5。
    原始 LeNet-5（输入 32x32，灰度）：
        C1: 6 个 5x5 卷积 -> 28x28   S2: 2x2 池化 -> 14x14
        C3: 16 个 5x5 卷积 -> 10x10  S4: 2x2 池化 -> 5x5
        C5: 120 全连接   F6: 84 全连接  输出: 10
    本实现支持多通道输入（CIFAR-10 用 in_channels=3），
    并对 28x28 输入做无 padding 卷积（24x24 -> 12x12 -> 8x8 -> 4x4）。
    通过 kernel_size / pool_type / pool_k / channels / act / stride 参数支持超参实验。
    """
    def __init__(self, in_channels=1, num_classes=10,
                 channels=(6, 16), in_h=32, in_w=32,
                 kernel_size=5, padding=0, pool_type="max", pool_k=2,
                 act="relu", stride=1):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[act]
        pool = nn.MaxPool2d if pool_type == "max" else nn.AvgPool2d

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size, stride=stride, padding=padding),
            act_fn(),
            pool(pool_k),
            nn.Conv2d(channels[0], channels[1], kernel_size, stride=stride, padding=padding),
            act_fn(),
            pool(pool_k),
        )
        # 通过一次虚拟前向计算推断展平维度，兼容不同输入尺寸/超参
        with torch.no_grad():
            x = torch.zeros(1, in_channels, in_h, in_w)
            flat = self.features(x).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Linear(flat, 120), act_fn(),
            nn.Linear(120, 84), act_fn(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


class MLP(nn.Module):
    """全连接神经网络（用于与 LeNet-5 对比、作为知识蒸馏学生）。"""
    def __init__(self, in_features, hidden=(512, 256), num_classes=10, act="relu"):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}[act]
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), act_fn()]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


# ============================================================
# 训练 / 评估
# ============================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def fit(model, train_loader, test_loader, epochs, lr=1e-3, momentum=0.9,
        log_every=1, label=""):
    """标准训练流程，返回历史记录。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        tl, ta = train_epoch(model, train_loader, criterion, optimizer)
        va = evaluate(model, test_loader)
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["test_acc"].append(va)
        if log_every and (ep % log_every == 0 or ep == 1):
            print(f"[{label}] epoch {ep}/{epochs} | train_loss={tl:.4f} train_acc={ta:.4f} test_acc={va:.4f}")
    return history


# ============================================================
# 绘图工具
# ============================================================
def save_fig(fig, fname):
    path = os.path.join(RESULTS_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存图表:", path)
    return path


def plot_history(histories, fname, title="训练曲线", ylabel="准确率",
                 key="test_acc", val_key=None):
    """histories: dict {标签: history}；画准确率/损失曲线。"""
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    for tag, h in histories.items():
        ax1.plot(h[key], label=f"{tag}({ylabel})")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel(ylabel)
    ax1.set_title(title); ax1.legend(); ax1.grid(alpha=0.3)
    return save_fig(fig, fname)


def plot_acc_loss(histories, fname, title=""):
    """双子图：左训练损失，右测试准确率。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for tag, h in histories.items():
        axes[0].plot(h["train_loss"], label=tag)
        axes[1].plot(h["test_acc"], label=tag)
    axes[0].set_title("训练损失"); axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("测试准确率"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    return save_fig(fig, fname)


def plot_image_grid(images, fname, nrow=10, title="", normalize=True, vmin=None, vmax=None):
    """images: numpy (N,H,W) 灰度图 或 (N,3,H,W)。"""
    if images.ndim == 3:
        images = images[:, None, :, :]
    t = torch.from_numpy(images).float()
    grid = torchvision.utils.make_grid(t, nrow=nrow, normalize=normalize, range=(vmin, vmax))
    fig, ax = plt.subplots(figsize=(nrow * 1.2, (images.shape[2] / images.shape[3]) * nrow * 1.2))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.axis("off")
    if title:
        ax.set_title(title)
    return save_fig(fig, fname)


def log_to_file(lines, fname):
    """把文本行追加写入 results/ 下文件。"""
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, "a", encoding="utf-8") as f:
        f.write(lines + "\n")
    return path
