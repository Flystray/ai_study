# -*- coding: utf-8 -*-
"""汇总 YOLOv3 检测结果：打印每张测试图的检测类别明细，并拼接效果图。"""
import os, sys, json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp8_yolov3 import load_yolo, detect, W, IMG_DIR
from common import RESULTS_DIR

CONF_THRESH = 0.4

net, class_names, out_layers = load_yolo()
test_imgs = [os.path.join(IMG_DIR, f) for f in sorted(os.listdir(IMG_DIR))
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]

summary = {}
rows = []
for p in test_imgs:
    img, _, picks = detect(net, out_layers, class_names, p)
    detail = []
    for cls_id, conf, (x, y, bw, bh) in picks:
        if conf >= CONF_THRESH:
            detail.append({"class": class_names[cls_id], "conf": round(float(conf), 3)})
    summary[os.path.basename(p)] = detail
    print(f"{os.path.basename(p):32s} -> {len(detail)} 个目标: "
          + ", ".join(f"{d['class']}({d['conf']:.2f})" for d in detail) if detail else "无")

# 拼接 6 张效果图
with open(os.path.join(RESULTS_DIR, "exp8_yolo_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
ax = axes.flat
shown = 0
for p in test_imgs:
    if shown >= 6:
        break
    img, _, picks = detect(net, out_layers, class_names, p)
    picks = [(cid, c, b) for cid, c, b in picks if c >= CONF_THRESH]
    if not picks:
        continue
    # 画框（BGR->RGB）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for cls_id, conf, (x, y, bw, bh) in picks:
        cv2.rectangle(img_rgb, (x, y), (x + bw, y + bh), (0, 200, 0), 3)
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.putText(img_rgb, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    ax[shown].imshow(img_rgb)
    ax[shown].axis("off")
    ax[shown].set_title(os.path.basename(p).replace(".jpg", ""), fontsize=9)
    shown += 1
fig.suptitle("YOLOv3 目标检测效果（COCO 80 类，OpenCV DNN 推理）", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, "exp8_yolov3_results.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n已保存: results/exp8_yolov3_results.png")
