# -*- coding: utf-8 -*-
"""
实验 1：LeNet-5 vs 全连接神经网络（MLP）
数据集：MNIST-digits / MNIST-Fashion / CIFAR-10
目标：验证 CNN（LeNet-5）相对全连接网络在图像分类上的效果优势。
"""
import os, sys, time, json
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, load_dataset, LeNet5, MLP,
                    fit, evaluate, count_params, plot_acc_loss, RESULTS_DIR)

set_seed(42)

# (数据集名, epoch 数, 学习率)
# Fashion/CIFAR 之前未充分收敛，已增大 epoch
CONFIGS = [
    ("mnist",   8, 1e-2),    # MNIST 8 epoch 已收敛（98.76%）
    ("fashion", 15, 1e-2),
    ("cifar10", 20, 1e-2),
]
import sys
# 支持命令行指定数据集：python exp1_lenet_vs_fnn.py mnist fashion
if len(sys.argv) > 1:
    CONFIGS = [c for c in CONFIGS if c[0] in sys.argv[1:]]

summary = {}
log_lines = []

for dname, epochs, lr in CONFIGS:
    print("=" * 60)
    print(f"数据集: {dname}")
    train_loader, test_loader, in_ch, h, w, ncls, classes = load_dataset(dname)
    in_features = in_ch * h * w

    # ---- LeNet-5 ----
    lenet = LeNet5(in_channels=in_ch, num_classes=ncls, in_h=h, in_w=w).to(DEVICE)
    # ---- MLP（隐藏层按输入规模调整，保证对比有讨论空间）----
    hidden = (200, 100) if dname != "cifar10" else (512, 256)
    mlp = MLP(in_features, hidden=hidden, num_classes=ncls).to(DEVICE)
    n_lenet, n_mlp = count_params(lenet), count_params(mlp)
    print(f"LeNet-5 参数量: {n_lenet:,} | MLP 参数量: {n_mlp:,}")

    t0 = time.time()
    print(f"[{dname}] 训练 LeNet-5 ...")
    h_lenet = fit(lenet, train_loader, test_loader, epochs, lr=lr, label=f"{dname}/LeNet5")
    acc_lenet = h_lenet["test_acc"][-1]

    print(f"[{dname}] 训练 MLP ...")
    h_mlp = fit(mlp, train_loader, test_loader, epochs, lr=lr, label=f"{dname}/MLP")
    acc_mlp = h_mlp["test_acc"][-1]

    # 对 MNIST 系再评估一次（确认无过拟合差异）
    final_acc_lenet = evaluate(lenet, test_loader)
    final_acc_mlp = evaluate(mlp, test_loader)

    cost = time.time() - t0
    print(f"[{dname}] 最终 LeNet-5 acc={final_acc_lenet:.4f} | MLP acc={final_acc_mlp:.4f} | 耗时 {cost:.0f}s")

    # 保存训练曲线
    plot_acc_loss(
        {f"LeNet-5": h_lenet, "MLP": h_mlp},
        f"exp1_{dname}_curve.png",
        title=f"LeNet-5 vs MLP on {dname}",
    )

    summary[dname] = {
        "lenet5_acc": final_acc_lenet, "mlp_acc": final_acc_mlp,
        "lenet5_params": n_lenet, "mlp_params": n_mlp, "cost_s": cost,
    }
    log_lines.append(
        f"{dname}\tLeNet5={final_acc_lenet:.4f}\tMLP={final_acc_mlp:.4f}"
        f"\tLeNet5_params={n_lenet}\tMLP_params={n_mlp}"
    )
    # 保存模型权重，供实验 3（知识蒸馏）复用
    if dname in ("mnist", "fashion"):
        torch.save(lenet.state_dict(), os.path.join(RESULTS_DIR, f"exp3_teacher_lenet_{dname}.pth"))

with open(os.path.join(RESULTS_DIR, "exp1_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
with open(os.path.join(RESULTS_DIR, "exp1_result.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))

print("实验 1 完成，结果见 results/exp1_*")
