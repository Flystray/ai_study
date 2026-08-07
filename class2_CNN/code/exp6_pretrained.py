# -*- coding: utf-8 -*-
"""
实验 6：预训练主流 CNN 在 ImageNet 上的验证
模型：AlexNet / VGG-16 / ResNet-50（torchvision 预训练权重）
数据：ImageNet 验证集（目录结构 data/imagenet/val/<synset>/xxx.JPEG，ImageFolder 格式）
      —— 由用户自行下载后放入 data/imagenet/val。
脚本在数据集不可用时会自动降级为"模型结构/参数量/FLOPs 对比"分析，
并在给出数据后直接输出 top-1 / top-5 准确率。
"""
import os, sys, json, time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, RESULTS_DIR, PROJECT_ROOT

IMAGENET_VAL = os.path.join(PROJECT_ROOT, "data", "imagenet", "val")
BATCH = 64

TRANSFORM = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_models(pretrained=True):
    """pretrained=True 加载 ImageNet 预训练权重（需下载）；
       pretrained=False 用随机初始化构建模型，仅用于参数量/FLOPs 结构分析。"""
    weights = "IMAGENET1K_V1" if pretrained else None
    return {
        "AlexNet": torchvision.models.alexnet(weights=weights),
        "VGG-16": torchvision.models.vgg16(weights=weights),
        "ResNet-50": torchvision.models.resnet50(weights=weights),
    }


def compute_flops(model, input_size=(1, 3, 224, 224)):
    """粗略统计 conv/linear 层的乘加次数（FLOPs ≈ 2×MACs）。"""
    flops = 0
    hooks = []
    def count_hook(module, inp, out):
        nonlocal flops
        x = inp[0]
        if isinstance(module, (nn.Conv2d,)):
            c_out = module.out_channels
            k = module.kernel_size[0] * module.kernel_size[1]
            flops += c_out * k * out.shape[-2] * out.shape[-1] * x.shape[1]
        elif isinstance(module, nn.Linear):
            flops += module.out_features * module.in_features
    for m in model.modules():
        hooks.append(m.register_forward_hook(count_hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(*input_size))
    for h in hooks:
        h.remove()
    return flops * 2


@torch.no_grad()
def evaluate_imagenet(model, loader):
    model.eval()
    top1 = top5 = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        pred = out.topk(5, 1).indices
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y[:, None]).sum().item()
        total += y.size(0)
    return top1 / total, top5 / total


def build_subset_loader(dataset, per_class):
    """每类取前 per_class 张，返回 (DataLoader, 子集索引列表)。"""
    idxs = []
    per = {}
    for i, (_, y) in enumerate(dataset.samples):
        if per.get(y, 0) < per_class:
            idxs.append(i)
            per[y] = per.get(y, 0) + 1
    sub = torch.utils.data.Subset(dataset, idxs)
    return torch.utils.data.DataLoader(sub, batch_size=64, shuffle=False, num_workers=0), idxs


def show_predictions(model, dataset, idxs, n=6):
    """展示若干张图片及其 Top-5 预测。"""
    model.eval()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    ax = axes.flat
    shown = 0
    for j in range(0, len(idxs), len(idxs) // 6):
        if shown >= 6:
            break
        i = idxs[j]
        path, y = dataset.samples[i]
        img = Image.open(path).convert("RGB")
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        out = model(x)
        probs = torch.softmax(out, 1)[0]
        top5 = out[0].topk(5).indices.tolist()
        names = [dataset.classes[k].split(",")[0] for k in top5]
        confs = [probs[k].item() for k in top5]
        true_name = dataset.classes[y].split(",")[0]
        ax[shown].imshow(img)
        ax[shown].axis("off")
        txt = "真实: " + true_name + "\n"
        txt += "\n".join([f"  {j+1}. {n} ({c:.1%})" for j, (n, c) in enumerate(zip(names, confs))])
        ax[shown].set_title(txt, fontsize=8)
        shown += 1
    fig.suptitle("ImageNet 预测结果示例（Top-5）", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp6_imagenet_predictions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存预测示例: results/exp6_imagenet_predictions.png")


def model_analysis(models):
    """不依赖数据：参数量、FLOPs、理论存储分析。"""
    rows = {}
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        flops = compute_flops(model)
        # 存储大小（fp32）
        mb = n_params * 4 / 1e6
        rows[name] = {"params": n_params, "flops": flops, "size_MB": mb}
        print(f"{name:10s} | 参数 {n_params/1e6:8.2f}M | FLOPs {flops/1e9:7.2f}G | 存储 {mb:6.1f} MB")
    return rows


def plot_comparison(rows, suffix=""):
    names = list(rows.keys())
    params = [rows[n]["params"] / 1e6 for n in names]
    flops = [rows[n]["flops"] / 1e9 for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, params, color="#4c8bf5")
    axes[0].set_title("参数量 (Million)"); axes[0].grid(alpha=0.3, axis="y")
    axes[1].bar(names, flops, color="#f5a742")
    axes[1].set_title("单张 224x224 推理 FLOPs (G)"); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"exp6_model_comparison{suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存: results/exp6_model_comparison.png")


def main():
    # 支持 --analysis-only：仅做结构分析（不下载权重/不推理 ImageNet）
    analysis_only = "--analysis-only" in sys.argv
    if analysis_only:
        print("结构分析模式（随机初始化模型，不下载权重）...")
        models = build_models(pretrained=False)
        rows = model_analysis(models)
        plot_comparison(rows)
        with open(os.path.join(RESULTS_DIR, "exp6_pretrained.json"), "w", encoding="utf-8") as f:
            json.dump({"model_analysis": rows, "imagenet_result": None}, f, ensure_ascii=False, indent=2)
        print("实验 6（结构分析）完成。需要 ImageNet 准确率时：\n"
              "  1) 运行 python exp6_pretrained.py 下载预训练权重；\n"
              "  2) 将 ImageNet 验证集放到 data/imagenet/val 后再次运行。")
        return

    print("加载预训练模型（首次运行会下载权重，需联网）...")
    models = build_models()
    rows = model_analysis(models)
    plot_comparison(rows)

    if not os.path.isdir(IMAGENET_VAL):
        print(f"\n[提示] 未找到 ImageNet 验证集: {IMAGENET_VAL}")
        print("请将验证集按 data/imagenet/val/<类别子目录>/<图片> 的结构放置后，重新运行本脚本。")
        print("（ImageNet 原始验证集需在 https://image-net.org 申请下载）")
        with open(os.path.join(RESULTS_DIR, "exp6_pretrained.json"), "w", encoding="utf-8") as f:
            json.dump({"model_analysis": rows, "imagenet_result": None}, f, ensure_ascii=False, indent=2)
        return

    dataset = torchvision.datasets.ImageFolder(IMAGENET_VAL, transform=TRANSFORM)
    per_class = None
    if "--subset-per-class" in sys.argv:
        per_class = int(sys.argv[sys.argv.index("--subset-per-class") + 1])
    if per_class:
        loader, idxs = build_subset_loader(dataset, per_class)
        print(f"使用子集测试：每类 {per_class} 张，共 {len(idxs)} 张（完整验证集为 {len(dataset)} 张）")
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=False, num_workers=0)
        idxs = list(range(len(dataset)))
        print(f"ImageNet 验证集样本数: {len(dataset)}")

    results = {}
    for i, (name, model) in enumerate(models.items()):
        model = model.to(DEVICE)
        t0 = time.time()
        acc1, acc5 = evaluate_imagenet(model, loader)
        cost = time.time() - t0
        print(f"{name:10s} | Top-1 {acc1:.4f} | Top-5 {acc5:.4f} | 耗时 {cost:.0f}s")
        results[name] = {"top1": acc1, "top5": acc5, "cost_s": cost, "n": len(idxs)}
        if i == 0:
            show_predictions(model, dataset, idxs, n=6)

    # 柱状图
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(results)))
    ax.bar([i - 0.18 for i in x], [results[n]["top1"] for n in results], 0.36, label="Top-1", color="#4c8bf5")
    ax.bar([i + 0.18 for i in x], [results[n]["top5"] for n in results], 0.36, label="Top-5", color="#f5a742")
    ax.set_xticks(x); ax.set_xticklabels(list(results))
    ax.set_title("ImageNet 验证集准确率对比"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    for i, n in enumerate(results):
        ax.text(i - 0.18, results[n]["top1"] + 0.01, f'{results[n]["top1"]:.3f}', ha="center", fontsize=9)
        ax.text(i + 0.18, results[n]["top5"] + 0.01, f'{results[n]["top5"]:.3f}', ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp6_imagenet_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "exp6_pretrained.json"), "w", encoding="utf-8") as f:
        json.dump({"model_analysis": rows, "imagenet_result": results}, f, ensure_ascii=False, indent=2)
    print("实验 6 完成，结果见 results/exp6_*")


if __name__ == "__main__":
    main()
