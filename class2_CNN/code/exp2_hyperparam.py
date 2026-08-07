# -*- coding: utf-8 -*-
"""
实验 2：LeNet-5 超参数对比（在 MNIST 上）
对比维度：卷积核大小 / 池化种类 / 池化大小 / 通道个数 / 激活函数 / stride。
设计原则：对比某一超参时，其余保持基线配置，且用 padding=k//2 使特征图尺寸一致，
          从而让差异只来自被考察的超参（保证对比的科学性）。
基线：kernel=5, padding=2, max pool(2), channels=(6,16), relu, stride=1
"""
import os, sys, json, time
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, load_dataset, LeNet5,
                    fit, evaluate, count_params, RESULTS_DIR)

set_seed(42)

EPOCHS = 1 if os.environ.get("SMOKE") else 4   # 超参对比用较少 epoch 观察趋势，兼顾 CPU 时间
LR = 1e-2
DATASET = "mnist"

def run_config(tag, **kw):
    print("=" * 50)
    print(f"[{tag}] {kw}")
    model = LeNet5(in_channels=1, in_h=28, in_w=28, **kw).to(DEVICE)
    h = fit(model, train_loader, test_loader, EPOCHS, lr=LR, label=tag)
    return h, count_params(model)

train_loader, test_loader, *_ = load_dataset(DATASET)

results = {}          # tag -> 最终测试准确率
curves = {}           # tag -> history

BASE = dict(kernel_size=5, padding=2, pool_type="max", pool_k=2,
            channels=(6, 16), act="relu", stride=1)

# ---------- 1. 卷积核大小（保持特征图尺寸一致：padding=k//2）----------
for k in (3, 5, 7):
    tag = f"kernel{k}"
    curves[tag], _ = run_config(tag, **{**BASE, "kernel_size": k, "padding": k // 2})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 2. 池化种类 ----------
for pt in ("max", "avg"):
    tag = f"pool_{pt}"
    curves[tag], _ = run_config(tag, **{**BASE, "pool_type": pt})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 3. 池化大小 ----------
for pk in (2, 3):
    tag = f"poolk{pk}"
    curves[tag], _ = run_config(tag, **{**BASE, "pool_k": pk})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 4. 通道个数 ----------
for ch in ((6, 16), (12, 32)):
    tag = f"channels_{ch[0]}-{ch[1]}"
    curves[tag], _ = run_config(tag, **{**BASE, "channels": ch})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 5. 激活函数 ----------
for act in ("relu", "tanh", "sigmoid"):
    tag = f"act_{act}"
    curves[tag], _ = run_config(tag, **{**BASE, "act": act})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 6. stride（stride=2 时特征图变小，仍可训练）----------
for s in (1, 2):
    tag = f"stride{s}"
    curves[tag], _ = run_config(tag, **{**BASE, "stride": s})
    results[tag] = curves[tag]["test_acc"][-1]

# ---------- 汇总与绘图 ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
groups = {
    "卷积核大小": ["kernel3", "kernel5", "kernel7"],
    "池化种类": ["pool_max", "pool_avg"],
    "池化大小": ["poolk2", "poolk3"],
    "通道个数": ["channels_6-16", "channels_12-32"],
    "激活函数": ["act_relu", "act_tanh", "act_sigmoid"],
    "stride": ["stride1", "stride2"],
}
for ax, (gname, tags) in zip(axes.flat, groups.items()):
    for t in tags:
        ax.plot(range(1, EPOCHS + 1), curves[t]["test_acc"], marker="o", label=t)
    ax.set_title(gname); ax.set_xlabel("Epoch"); ax.set_ylabel("测试准确率")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle(f"LeNet-5 超参数对比（{DATASET}，{EPOCHS} epochs）", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "exp2_hyperparam_curves.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

with open(os.path.join(RESULTS_DIR, "exp2_hyperparam.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n超参数对比最终测试准确率：")
for k, v in results.items():
    print(f"  {k:20s} -> {v:.4f}")
print("图表已保存: results/exp2_hyperparam_curves.png")
