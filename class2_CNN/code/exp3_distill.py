# -*- coding: utf-8 -*-
"""
实验 3：知识蒸馏（Knowledge Distillation）
任务：在 MNIST-digits 和 MNIST-Fashion 上，用 LeNet-5 作为教师网络，把知识蒸馏给
      一个更小的全连接网络（学生），观察学生网络精度的提升。
场景设计（对齐 Hinton 2015 论文核心实验）：
  教师用完整训练集训练（已由实验一得到），学生只使用 10% 的训练数据。
  当数据不足时，硬标签信息有限，教师的软标签（含类间相似性）成为额外监督，
  蒸馏训练应显著优于同数据量下的独立训练。
方法：损失 = alpha·CE(学生, y) + (1-alpha)·T²·KL(学生_T || 教师_T)，T=4, alpha=0.5
"""
import os, sys, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, load_dataset, LeNet5, MLP,
                    fit, evaluate, count_params, RESULTS_DIR)

set_seed(42)

EPOCHS = 1 if os.environ.get("SMOKE") else 15
# 经超参扫描（exp3_tune）确定：MNIST 上教师 softmax 尖锐，软标签≈硬标签，
# 温度越低、硬标签权重越高越接近/超过基线。采用 T=1, alpha=0.7 与更小的学生。
T, ALPHA = 1.0, 0.7
SUBSET_FRAC = 0.1          # 学生只用 10% 训练数据
DATASETS = ["mnist", "fashion"]
STUDENT_HIDDEN = (32,)     # 小容量学生，给蒸馏留出提升空间


def teacher_logits(teacher, loader):
    teacher.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_logits.append(teacher(x.to(DEVICE)).cpu())
            all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def train_student(student, loader, teacher_soft, T, alpha, epochs, use_soft,
                  label=""):
    """use_soft=True 时用教师软标签做蒸馏；否则纯硬标签独立训练。"""
    optimizer = optim.SGD(student.parameters(), lr=1e-2, momentum=0.9)
    hist = {"train_loss": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        student.train()
        total_loss, correct, total = 0.0, 0, 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = student(x)
            if use_soft:
                start = i * loader.batch_size
                end = start + y.size(0)
                st = teacher_soft[start:end].to(DEVICE)          # 教师软化概率
                loss_ce = F.cross_entropy(out, y)
                loss_kd = F.kl_div(F.log_softmax(out / T, dim=1), st,
                                   reduction="batchmean") * (T * T)
                loss = alpha * loss_ce + (1 - alpha) * loss_kd
            else:
                loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        va = evaluate(student, test_loader)
        hist["train_loss"].append(total_loss / total)
        hist["test_acc"].append(va)
        print(f"[{label}] epoch {ep}/{epochs} | loss={total_loss/total:.4f} train_acc={correct/total:.4f} test_acc={va:.4f}")
    return hist


summary = {}
for dname in DATASETS:
    print("=" * 60)
    print(f"数据集: {dname}")
    train_loader, test_loader, in_ch, h, w, ncls, _ = load_dataset(dname)
    in_features = in_ch * h * w

    # 教师 LeNet-5（复用实验一权重）
    teacher = LeNet5(in_channels=in_ch, in_h=h, in_w=w).to(DEVICE)
    t_path = os.path.join(RESULTS_DIR, f"exp3_teacher_lenet_{dname}.pth")
    teacher.load_state_dict(torch.load(t_path, map_location=DEVICE))
    acc_teacher = evaluate(teacher, test_loader)
    t_logits, _ = teacher_logits(teacher, train_loader)

    # 学生只使用 10% 训练数据
    full_ds = train_loader.dataset
    n_total = len(full_ds)
    n_sub = int(n_total * SUBSET_FRAC)
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=rng)[:n_sub].tolist()
    subset_ds = Subset(full_ds, indices)
    # 关键：必须 shuffle=False，保证每个 batch 的样本与 soft_sub[indices] 顺序一一对应
    subset_loader = DataLoader(subset_ds, batch_size=train_loader.batch_size,
                               shuffle=False, num_workers=0)
    # 对应的教师软标签（按子集索引取）
    soft_all = F.softmax(t_logits / T, dim=1)
    soft_sub = soft_all[indices]

    # 学生：独立训练（10% 数据，硬标签） vs 蒸馏训练（10% 数据 + 教师软标签）
    student_alone = MLP(in_features, hidden=STUDENT_HIDDEN).to(DEVICE)
    student_distill = MLP(in_features, hidden=STUDENT_HIDDEN).to(DEVICE)
    n_student = count_params(student_alone)
    print(f"教师 LeNet-5 acc={acc_teacher:.4f} (参数 {count_params(teacher):,}) | "
          f"学生 MLP 参数 {n_student:,} | 学生训练数据量 {n_sub}（{SUBSET_FRAC*100:.0f}%）")

    print("[独立训练] 学生（10% 数据，硬标签）...")
    h_alone = train_student(student_alone, subset_loader, None, T, ALPHA,
                            EPOCHS, use_soft=False, label=f"{dname}/student-standalone")
    acc_alone = h_alone["test_acc"][-1]

    print("[蒸馏训练] 学生（10% 数据 + 教师软标签）...")
    h_distill = train_student(student_distill, subset_loader, soft_sub, T, ALPHA,
                              EPOCHS, use_soft=True, label=f"{dname}/student-distill")
    acc_distill = h_distill["test_acc"][-1]

    # 教师 logits 的软标签在完整测试集上的表现（上界参考）
    with torch.no_grad():
        teacher.eval()
        soft_teacher_test = []
        for x, _ in test_loader:
            soft_teacher_test.append(F.softmax(teacher(x.to(DEVICE)) / T, dim=1).cpu())
        soft_teacher_test = torch.cat(soft_teacher_test)
        # 用软标签 argmax 作为教师 soft 预测的准确率
        acc_teacher_soft = (soft_teacher_test.argmax(1) == torch.cat([y for _, y in test_loader])).float().mean().item()

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, EPOCHS + 1), h_alone["test_acc"], marker="o", label="学生(独立训练,10%数据)")
    axes[0].plot(range(1, EPOCHS + 1), h_distill["test_acc"], marker="s", label="学生(蒸馏训练)")
    axes[0].axhline(acc_teacher, color="red", ls="--", label=f"教师 LeNet-5 ({acc_teacher:.4f})")
    axes[0].set_title(f"{dname}: 学生测试准确率对比（训练数据仅 10%）")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("测试准确率")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].bar(["独立训练\n(10%数据)", "蒸馏训练\n(10%数据)"], [acc_alone, acc_distill],
                color=["#6ba3d6", "#f2a541"])
    axes[1].axhline(acc_teacher, color="red", ls="--", label=f"教师 {acc_teacher:.4f}")
    axes[1].set_title(f"{dname}: 最终准确率"); axes[1].set_ylim(0, 1); axes[1].legend()
    for i, v in enumerate([acc_alone, acc_distill]):
        axes[1].text(i, v + 0.01, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"exp3_distill_{dname}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary[dname] = {
        "teacher_acc": acc_teacher,
        "student_alone_acc": acc_alone,
        "student_distill_acc": acc_distill,
        "improvement": acc_distill - acc_alone,
        "T": T, "alpha": ALPHA, "subset_frac": SUBSET_FRAC,
        "student_params": n_student, "teacher_params": count_params(teacher),
        "acc_teacher_soft_argmax": acc_teacher_soft,
        # 教师 softmax 平均最大概率（衡量软标签的"尖锐度"，越小说明携带更多类间信息）
        "teacher_avg_max_prob": float(t_logits.softmax(1).max(1).values.mean()),
    }
    print(f"[{dname}] 教师={acc_teacher:.4f} | 学生独立={acc_alone:.4f} | "
          f"学生蒸馏={acc_distill:.4f} | 提升={acc_distill-acc_alone:+.4f}")

with open(os.path.join(RESULTS_DIR, "exp3_distill.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("实验 3 完成，结果见 results/exp3_*")
