# -*- coding: utf-8 -*-
"""
AE 可视化演示：
1. 训练一个 CNN-AE（MNIST，短训几个 epoch）；
2. 对几张图打印潜在向量 z（64 个值）；
3. 计算不同图的 z 距离（同类近、异类远）；
4. 潜在空间插值：在两个数字的 z 之间线性插值 → 解码 → 展示"7→3"的平滑渐变。
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, load_dataset, RESULTS_DIR
from exp4_ae import ConvAE, MLPAE, train_ae

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 1. 训练 CNN-AE（MNIST，6 epochs，演示够用；权重已缓存则直接加载）----
train_loader, test_loader, in_ch, h, w, *_ = load_dataset("mnist", batch_size=128, normalize=False)
CKPT = os.path.join(RESULTS_DIR, "demo_cnn_ae.ckpt")
ae = ConvAE(in_channels=in_ch).to(DEVICE)
if os.path.exists(CKPT):
    ae.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print("加载已训练的 AE 权重（跳过训练）")
else:
    train_ae(ae, train_loader, epochs=6, label="demo/cnn-ae")
    torch.save(ae.state_dict(), CKPT)
    print("AE 训练完成，权重已缓存")
ae.eval()

# 取 3 张图：两张"7" + 一张"3"
def take(digit, n=2):
    got = []
    for xb, yb in test_loader:
        for i in range(yb.size(0)):
            if yb[i].item() == digit:
                got.append(xb[i:i+1].to(DEVICE))
            if len(got) == n:
                return got
    return got

x7a, x7b = take(7, 2)   # 两个"7"
x3a = take(3, 1)[0]      # 一个"3"

# ---- 2. 前向拿潜在向量 z 和重建 ----
def get_z(x):
    with torch.no_grad():
        h = ae.encoder(x).view(x.size(0), -1)
        return ae.fc(h)
def reconstruct(x):
    with torch.no_grad():
        return ae(x).cpu().numpy()

z7a, z7b, z3 = get_z(x7a), get_z(x7b), get_z(x3a)
print("=" * 70)
print("潜在向量 z（64 个数字，即图像被压缩成的'摘要'）")
print("=" * 70)
for name, z in [("7-a", z7a), ("7-b", z7b), ("3", z3)]:
    vals = z[0].cpu().numpy()
    print(f"\n【{name}】")
    # 按 8x8 打印，方便看整体
    for r in range(8):
        print("   " + "  ".join(f"{v:6.2f}" for v in vals[r*8:(r+1)*8]))

# ---- 3. z 之间的距离 ----
def dist(a, b):
    return float(((a - b) ** 2).sum().sqrt())
print("\n" + "=" * 70)
print("z 之间的距离（欧氏距离）")
print("=" * 70)
print(f"  两个'7'的 z 距离 : {dist(z7a, z7b):.3f}   ← 同类，应较小")
print(f"  '7'和'3'的 z 距离 : {dist(z7a, z3):.3f}   ← 异类，应较大")

# ---- 4. 潜在空间插值：7 -> 3 ----
interp_zs = [(1 - t) * z7a + t * z3 for t in [0, 0.2, 0.4, 0.6, 0.8, 1.0]]
imgs = []
for z in interp_zs:
    with torch.no_grad():
        d = ae.fc_back(z).view(-1, 128, 4, 4)
        rec = ae.out(ae.crop(ae.decoder(d)))
        imgs.append(rec[0, 0].cpu().numpy())

# ---- 画图 ----
fig, axes = plt.subplots(2, 6, figsize=(18, 6))
fig.subplots_adjust(wspace=0.25, hspace=0.45)
row_labels = ["原图", "重建"]

# 第一行：3 组「原图 | 重建」
pairs = [(x7a, "数字 7 (a)"), (x7b, "数字 7 (b)"), (x3a, "数字 3")]
for j, (x, name) in enumerate(pairs):
    axes[0, 2*j].imshow(x[0, 0].cpu().numpy(), cmap="gray")
    axes[0, 2*j].set_title("原图", fontsize=12)
    axes[0, 2*j].axis("off")
    axes[0, 2*j+1].imshow(reconstruct(x)[0, 0], cmap="gray")
    axes[0, 2*j+1].set_title("重建", fontsize=12)
    axes[0, 2*j+1].axis("off")
    # 每组上方标注：是哪个数字
    axes[0, 2*j].text(0.5, 1.35, name, transform=axes[0, 2*j].transAxes,
                      ha="center", fontsize=13, fontweight="bold")

# 第二行：潜在空间插值 7 -> 3
interp_labels = [f"t=0.0", "t=0.2", "t=0.4", "t=0.6", "t=0.8", "t=1.0"]
for j, img in enumerate(imgs):
    axes[1, j].imshow(img, cmap="gray")
    axes[1, j].set_title(interp_labels[j], fontsize=12)
    axes[1, j].axis("off")

fig.suptitle("AutoEncoder 演示\n上排：原图与 AE 重建对比  |  下排：潜在空间插值（从 z(7) 渐变到 z(3)）",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(RESULTS_DIR, "exp4_visualize.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n可视化图已保存:", out)
