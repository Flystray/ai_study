# -*- coding: utf-8 -*-
"""展示 data/ 下各数据集（MNIST / FashionMNIST / CIFAR-10）的部分样本图片。
保存为网格图到 results/dataset_preview/ 供查看。"""
import os, sys
import numpy as np
import torchvision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RESULTS_DIR, DATA_DIR

OUT = os.path.join(RESULTS_DIR, "dataset_preview")
os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

SHOW_PER_CLASS = 5   # 每类展示几张


def show_grayscale(ds, classes, title, fname):
    """ds: torchvision 数据集（transform=ToTensor），展示每类若干样本。
    横排网格：行=样本序号，列=类别，便于一行横向对比所有类别。"""
    per = {c: 0 for c in range(len(classes))}
    imgs = []
    for x, y in ds:
        if per[y] < SHOW_PER_CLASS:
            imgs.append((x.squeeze().numpy(), y))
            per[y] += 1
        if all(v >= SHOW_PER_CLASS for v in per.values()):
            break
    n_cls = len(classes)
    fig, axes = plt.subplots(SHOW_PER_CLASS, n_cls, figsize=(n_cls * 1.3, SHOW_PER_CLASS * 1.3))
    for r in range(SHOW_PER_CLASS):          # 行 = 样本序号
        for c in range(n_cls):               # 列 = 类别
            ax = axes[r, c]
            col = [im for im in imgs if im[1] == c]
            ax.imshow(col[r][0], cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(classes[c], fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", p)


def show_color(ds, classes, title, fname):
    """CIFAR-10 彩色数据集。横排网格：行=样本序号，列=类别。"""
    per = {c: 0 for c in range(len(classes))}
    imgs = []
    for x, y in ds:
        if per[y] < SHOW_PER_CLASS:
            imgs.append((x.permute(1, 2, 0).numpy(), y))   # C,H,W -> H,W,C
            per[y] += 1
        if all(v >= SHOW_PER_CLASS for v in per.values()):
            break
    n_cls = len(classes)
    fig, axes = plt.subplots(SHOW_PER_CLASS, n_cls, figsize=(n_cls * 1.3, SHOW_PER_CLASS * 1.3))
    for r in range(SHOW_PER_CLASS):          # 行 = 样本序号
        for c in range(n_cls):               # 列 = 类别
            ax = axes[r, c]
            col = [im for im in imgs if im[1] == c]
            ax.imshow(col[r][0])
            ax.axis("off")
            if r == 0:
                ax.set_title(classes[c], fontsize=8)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("已保存:", p)


# MNIST
mnist = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=False,
                                   transform=torchvision.transforms.ToTensor())
show_grayscale(mnist, [str(i) for i in range(10)], "MNIST 手写数字数据集（每类5张）", "mnist_samples.png")

# Fashion-MNIST
fashion = torchvision.datasets.FashionMNIST(root=DATA_DIR, train=True, download=False,
                                            transform=torchvision.transforms.ToTensor())
fashion_classes = ["T恤/上衣", "裤子", "套衫", "裙子", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]
show_grayscale(fashion, fashion_classes, "Fashion-MNIST 服饰数据集（每类5张）", "fashion_samples.png")

# CIFAR-10
cifar = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False,
                                     transform=torchvision.transforms.ToTensor())
show_color(cifar, cifar.classes, "CIFAR-10 自然图像数据集（每类5张）", "cifar10_samples.png")

print("全部数据集预览图生成完毕，位于 results/dataset_preview/")
