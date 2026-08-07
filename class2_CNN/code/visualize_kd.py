# -*- coding: utf-8 -*-
"""
知识蒸馏可视化演示（不需要训练）：
1. 加载已训练好的教师 LeNet-5，对一张手写数字做一次前向；
2. 打印它的 logits，以及 softmax(T=1) 和 softmax(T=4) 的概率分布；
3. 可视化教师第一层卷积的 6 张特征图；
4. 对比"硬标签 vs 教师软标签"，直观展示"暗知识"。
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEVICE, load_dataset, LeNet5, RESULTS_DIR)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 加载教师
teacher = LeNet5(in_channels=1, in_h=28, in_w=28).to(DEVICE)
teacher.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "exp3_teacher_lenet_mnist.pth"),
                                   map_location=DEVICE))
teacher.eval()

# 取测试集里一张数字"7"（7 和 1 形状相似，适合展示暗知识）
train_loader, test_loader, *_ = load_dataset("mnist", batch_size=64)
target_digit = 7
for xb, yb in test_loader:
    idx = (yb == target_digit).nonzero()[0].item()
    x, y = xb[idx:idx+1].to(DEVICE), yb[idx].item()
    break

print("=" * 70)
print(f"输入图片：真实标签 = {y}（手写数字）")
print("=" * 70)

# ---- 一次性拿到 logits + 第一层特征图 ----
hook_out = {}
def grab(name):
    def hook(m, inp, out):
        hook_out[name] = out.detach()
    return hook

# 注册钩子抓取第一层卷积输出
teacher.features[0].register_forward_hook(grab("conv1"))

with torch.no_grad():
    logits = teacher(x)                                  # (1, 10)
    feat1 = hook_out["conv1"]                            # (1, 6, 24, 24)

print("\n【教师的原始输出 logits】")
print("   ", [f"{v:.2f}" for v in logits[0].tolist()])

# ---- 不同温度下的 softmax ----
for T in (1.0, 4.0):
    p = F.softmax(logits / T, dim=1)[0].tolist()
    print(f"\n【softmax(T={T})  软化后的概率分布】")
    row = "    " + "  ".join(f"{v:.3f}" for v in p)
    print(row)
    if T == 1.0:
        # 硬标签（真实 one-hot）
        hard = F.one_hot(torch.tensor(y), 10).numpy().tolist()
        print("    硬标签(y): " + "  ".join(f"{v:.3f}" for v in hard))

# ---- 对比：硬标签 vs 教师软标签 vs 学生理想分布 ----
p1 = F.softmax(logits / 1.0, dim=1)[0].cpu()
p4 = F.softmax(logits / 4.0, dim=1)[0].cpu()
print("\n【暗知识在哪？】 对比 1 和 7 两个数字的概率")
print(f"   硬标签说：'1'=0.0, '7'=1.0  （看不到 1 和 7 的关系）")
print(f"   教师软标签(T=4)说：'1'={p4[1]:.3f}, '7'={p4[7]:.3f}  "
      f"→ 7 也像 1（因为笔画相似），这就是硬标签没有的知识")

# ---- 画图 ----
classes = [str(i) for i in range(10)]
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
# 原始图
axes[0, 0].imshow(x[0, 0].cpu().numpy(), cmap="gray")
axes[0, 0].set_title(f"输入：手写数字 {y}")
axes[0, 0].axis("off")

# 概率分布对比图放第一行第二列（蓝色 T=1 / 黄色 T=4 / 灰色硬标签）
ax = axes[0, 1]
xpos = np.arange(10)
ax.bar(xpos - 0.25, F.one_hot(torch.tensor(y), 10).numpy(), width=0.2,
       color="gray", label="硬标签")
ax.bar(xpos, p1.numpy(), width=0.2, color="steelblue", label="教师 softmax(T=1)")
ax.bar(xpos + 0.25, p4.numpy(), width=0.2, color="orange", label="教师 softmax(T=4)")
ax.set_xticks(xpos); ax.set_xticklabels(classes)
ax.set_title("同一张图，教师输出的不同形式")
ax.legend(fontsize=7); ax.set_ylabel("概率")
ax.set_ylim(0, 1.05)

# 4 张第一层特征图放其余格子
for i, (r, c) in enumerate([(0, 2), (1, 0), (1, 1), (1, 2)]):
    ax = axes[r, c]
    ax.imshow(feat1[0, i].cpu().numpy(), cmap="gray")
    ax.set_title(f"卷积核{i+1}提取的特征")
    ax.axis("off")

fig.suptitle(f"知识蒸馏可视化：教师对数字 {y} 的'看法'（温度软化 + 特征图）", fontsize=13)
fig.tight_layout()
out = os.path.join(RESULTS_DIR, "exp3_visualize.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n可视化图已保存:", out)
