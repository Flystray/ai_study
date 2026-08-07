# -*- coding: utf-8 -*-
"""
蒸馏超参快速扫描（诊断）：在 MNIST 10% 数据子集上，扫描温度 T 与权重 alpha，
找到"蒸馏训练 > 独立训练"的有效配置，供 exp3 最终实验使用。
每个配置只训练 5 epochs，用于观察趋势。
"""
import os, sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, load_dataset, LeNet5, MLP,
                    evaluate, count_params, RESULTS_DIR)

set_seed(42)
SUBSET_FRAC = 0.1
EPOCHS = 5

train_loader, test_loader, in_ch, h, w, ncls, _ = load_dataset("mnist")
in_features = in_ch * h * w

# 教师
teacher = LeNet5(in_channels=in_ch, in_h=h, in_w=w).to(DEVICE)
teacher.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "exp3_teacher_lenet_mnist.pth"),
                                   map_location=DEVICE))
acc_teacher = evaluate(teacher, test_loader)
print(f"教师 acc={acc_teacher:.4f}")

with torch.no_grad():
    t_logits_all = []
    for x, _ in train_loader:
        t_logits_all.append(teacher(x.to(DEVICE)).cpu())
    t_logits_all = torch.cat(t_logits_all)

# 子集（10% 数据）
full_ds = train_loader.dataset
n_sub = int(len(full_ds) * SUBSET_FRAC)
indices = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(42))[:n_sub].tolist()
subset_ds = Subset(full_ds, indices)
subset_loader = DataLoader(subset_ds, batch_size=train_loader.batch_size,
                           shuffle=False, num_workers=0)
soft_all = F.softmax(t_logits_all / 4.0, dim=1)  # 先按 T=4 预计算教师分布（扫描时在循环内按各自 T 重新算）

# 基线：独立训练（硬标签）
def train_student(T, alpha, use_soft):
    student = MLP(in_features, hidden=(32,)).to(DEVICE)   # 更小的学生
    opt = optim.SGD(student.parameters(), lr=1e-2, momentum=0.9)
    for ep in range(EPOCHS):
        student.train()
        for i, (x, y) in enumerate(subset_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = student(x)
            if use_soft:
                start = i * subset_loader.batch_size
                end = start + y.size(0)
                st = F.softmax(t_logits_all[indices[start:end]] / T, dim=1).to(DEVICE)
                loss = alpha * F.cross_entropy(out, y) + (1 - alpha) * \
                    F.kl_div(F.log_softmax(out / T, dim=1), st, reduction="batchmean") * (T * T)
            else:
                loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
    return evaluate(student, test_loader)

print(f"独立训练(硬标签) baseline: {train_student(0, 0, use_soft=False):.4f}")
print("-" * 50)
for T in (1, 2, 4, 6):
    for alpha in (0.3, 0.5, 0.7):
        acc = train_student(T, alpha, use_soft=True)
        print(f"T={T}, alpha={alpha}: 蒸馏 acc={acc:.4f}")
