# -*- coding: utf-8 -*-
"""
AE 潜在空间可视化（重新设计，一张图只讲一件事）：
图1 exp4_latent_space.png：把 0-9 各 30 张图编码成 z，PCA 降到 2D 画散点，
    直观看到"同一数字聚成团、不同数字互相分开、7 和 3 相距较远"。
图2 exp4_interp.png：极简插值——左端真实"7"、右端真实"3"，中间是潜在空间
    插值解码的过渡帧，展示从 7 渐变到 3。
"""
import os, sys
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, load_dataset, RESULTS_DIR
from exp4_ae import ConvAE

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ---- 加载已缓存的 AE 权重 ----
train_loader, test_loader, in_ch, h, w, *_ = load_dataset("mnist", batch_size=128, normalize=False)
CKPT = os.path.join(RESULTS_DIR, "demo_cnn_ae.ckpt")
ae = ConvAE(in_channels=in_ch).to(DEVICE)
if os.path.exists(CKPT):
    ae.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    print("加载已训练的 AE 权重")
else:
    sys.exit("请先运行 visualize_ae.py 训练并缓存权重")
ae.eval()

def get_z(x):
    with torch.no_grad():
        return ae.fc(ae.encoder(x).view(x.size(0), -1))
def decode(z):
    with torch.no_grad():
        d = ae.fc_back(z).view(-1, 128, 4, 4)
        return ae.out(ae.crop(ae.decoder(d)))[0, 0].cpu().numpy()

# ---- 收集 0-9 各 30 张的 z ----
per = defaultdict(list)
for xb, yb in test_loader:
    for i in range(yb.size(0)):
        lab = yb[i].item()
        if len(per[lab]) < 30:
            per[lab].append(get_z(xb[i:i+1].to(DEVICE))[0].cpu().numpy())
    if all(len(v) >= 30 for v in per.values()):
        break

zs = np.array([z for lab in range(10) for z in per[lab]])
labels = np.array([lab for lab in range(10) for _ in per[lab]])

# ---- PCA 降维到 2D ----
centered = zs - zs.mean(0)
U, S, Vt = np.linalg.svd(centered, full_matrices=False)
z2d = centered @ Vt[:2].T

# ---- 图1：潜在空间分布 ----
colors = plt.cm.tab10(np.arange(10))
fig, ax = plt.subplots(figsize=(8, 6))
for lab in range(10):
    mask = labels == lab
    ax.scatter(z2d[mask, 0], z2d[mask, 1], s=12, c=[colors[lab]],
               label=f"{lab}", alpha=0.7)
# 高亮 7 和 3 的聚团中心
for lab, marker in [(7, "D"), (3, "s")]:
    cx, cy = z2d[labels == lab].mean(0)
    ax.scatter(cx, cy, marker=marker, s=120, edgecolors="black", c=[colors[lab]], zorder=5)
    ax.annotate(f"数字{lab}中心", (cx, cy), textcoords="offset points",
                xytext=(12, 12), fontsize=11, fontweight="bold")
ax.set_xlabel("PCA 第 1 主成分")
ax.set_ylabel("PCA 第 2 主成分")
ax.set_title("潜在空间分布：同一数字聚成团，不同数字互相分开\n（7 与 3 相距较远，说明 z 编码了内容语义）", fontsize=12)
ax.legend(title="数字", ncol=5, fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
out1 = os.path.join(RESULTS_DIR, "exp4_latent_space.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存:", out1)

# ---- 图2：极简插值 7 -> 3 ----
x7 = per[7][0]
x3 = per[3][0]
z7 = torch.tensor(x7, device=DEVICE).unsqueeze(0)
z3 = torch.tensor(x3, device=DEVICE).unsqueeze(0)
# 找 7 和 3 的原始图（从测试集）
real7 = real3 = None
for xb, yb in test_loader:
    for i in range(yb.size(0)):
        if real7 is None and yb[i].item() == 7:
            real7 = xb[i, 0].cpu().numpy()
        if real3 is None and yb[i].item() == 3:
            real3 = xb[i, 0].cpu().numpy()
    if real7 is not None and real3 is not None:
        break

ts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
frames = [decode((1 - t) * z7 + t * z3) for t in ts]

fig, axes = plt.subplots(1, 8, figsize=(15, 3))
# 两端放真实图作参照
axes[0].imshow(real7, cmap="gray"); axes[0].set_title("真实 7", fontsize=11); axes[0].axis("off")
axes[-1].imshow(real3, cmap="gray"); axes[-1].set_title("真实 3", fontsize=11); axes[-1].axis("off")
# 中间插值帧
for j, (t, frame) in enumerate(zip(ts, frames)):
    axes[j + 1].imshow(frame, cmap="gray")
    axes[j + 1].set_title(f"t={t:.1f}", fontsize=11)
    axes[j + 1].axis("off")
axes[3].annotate("在潜在空间中从 z(7) 逐步过渡到 z(3)", xy=(0.5, -0.35),
                 xycoords="axes fraction", ha="center", fontsize=12, fontweight="bold",
                 annotation_clip=False)
fig.suptitle("插值：在'7 的摘要'和'3 的摘要'之间一步一步走，每步解码成图", fontsize=13)
fig.tight_layout()
out2 = os.path.join(RESULTS_DIR, "exp4_interp.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存:", out2)
