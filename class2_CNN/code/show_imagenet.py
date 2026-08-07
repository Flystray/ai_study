# -*- coding: utf-8 -*-
"""展示 ImageNet 验证集的部分图片（按类别，每类显示数张）。"""
import os, sys, json, random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
VAL = os.path.join(DATA, "imagenet", "val")
MAP = os.path.join(DATA, "imagenet_devkit", "synset_map.json")
OUT = os.path.join(os.path.dirname(HERE), "results", "dataset_preview")

os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

with open(MAP) as f:
    synset_map = json.load(f)   # {id: [wnid, words]}

# 展示的类别：挑几个常见/有趣类别，或用前 N 个 synset
classes = sorted(os.listdir(VAL))
random.seed(42)
# 均匀抽样 12 个类别，覆盖不同视觉类别
sample = classes[:6] + classes[500:503] + classes[900:903]
SHOW = 4   # 每类显示 4 张

n_cls = len(sample)
fig, axes = plt.subplots(SHOW, n_cls, figsize=(n_cls * 2.2, SHOW * 2.2))
for c, wnid in enumerate(sample):
    d = os.path.join(VAL, wnid)
    files = sorted(os.listdir(d))[:SHOW]
    # 类别名（英文 words，截断显示）
    label = "unknown"
    for k, v in synset_map.items():
        if v[0] == wnid:
            label = v[1]
            break
    for r, f in enumerate(files):
        img = Image.open(os.path.join(d, f)).convert("RGB")
        ax = axes[r, c]
        ax.imshow(img)
        ax.axis("off")
        if r == 0:
            ax.set_title(f"{wnid}\n{label[:18]}", fontsize=7)
fig.suptitle("ImageNet 验证集样例（synset ID + 类别名）", fontsize=13)
fig.tight_layout()
p = os.path.join(OUT, "imagenet_samples.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print("已保存:", p)

# 同步生成缩小一半的版本（横版）
img = Image.open(p)
w, h = img.size
img.resize((w // 2, h // 2), Image.LANCZOS).save(os.path.join(OUT, "imagenet_samples_small.png"))
print("已保存: imagenet_samples_small.png")
