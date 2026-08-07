# -*- coding: utf-8 -*-
"""
选做实验 8：YOLOv3 目标检测测试
论文：YOLOv3: An Incremental Improvement (Redmon & Farhadi, 2018)
方法：OpenCV DNN 加载 Darknet 的 cfg+weights，对 COCO 测试图片做单阶段检测。
依赖：opencv-python（cv2）、yolov3.cfg / yolov3.weights / coco.names（用户下载）。
下载地址（Darknet 官网）：
    cfg:     https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    weights: https://pjreddie.com/media/files/yolov3.weights   (~236MB)
    names:   https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
用法：把上面三个文件放到 code/weights/ 下，把测试图片放到 results/test_images/，
     运行本脚本即可。
"""
import os, sys, time
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(HERE, "weights")
CFG = os.path.join(W, "yolov3.cfg")
WEIGHTS = os.path.join(W, "yolov3.weights")
NAMES = os.path.join(W, "coco.names")
IMG_DIR = os.path.join(os.path.dirname(HERE), "results", "test_images")
CONF_THRESH, NMS_THRESH = 0.5, 0.4


def load_yolo():
    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open(NAMES, "r") as f:
        class_names = [l.strip() for l in f.readlines()]
    layer_names = net.getLayerNames()
    out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, class_names, out_layers


def detect(net, out_layers, class_names, img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    boxes, confs, ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cls_id = int(np.argmax(scores))
            conf = float(scores[cls_id])
            if conf > CONF_THRESH:
                cx, cy, bw, bh = det[:4] * [w, h, w, h]
                boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
                confs.append(conf); ids.append(cls_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESH, NMS_THRESH)
    if len(idxs) == 0:
        return img, class_names, []
    idxs = idxs.flatten()
    picks = [(ids[i], confs[i], boxes[i]) for i in idxs]
    return img, class_names, picks


def draw(img, class_names, picks):
    colors = np.random.randint(0, 255, (len(class_names), 3)).tolist()
    for cls_id, conf, (x, y, bw, bh) in picks:
        label = f"{class_names[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x, y), (x + bw, y + bh), colors[cls_id], 2)
        cv2.putText(img, label, (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls_id], 2)
    return img


def main():
    for f in (CFG, WEIGHTS, NAMES):
        if not os.path.exists(f):
            print(f"[缺失] {f}\n请先下载 YOLOv3 的 cfg / weights / names 三个文件。")
            return
    net, class_names, out_layers = load_yolo()
    print("YOLOv3 加载成功，开始测试图片 ...")
    os.makedirs(IMG_DIR, exist_ok=True)
    test_imgs = [os.path.join(IMG_DIR, f) for f in os.listdir(IMG_DIR)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not test_imgs:
        print(f"[提示] 请把测试图片放到 {IMG_DIR} 目录。")
        return
    for p in test_imgs:
        t0 = time.time()
        img, names, picks = detect(net, out_layers, class_names, p)
        print(f"{os.path.basename(p)}: 检测到 {len(picks)} 个目标, 耗时 {time.time()-t0:.2f}s")
        result = draw(img, names, picks)
        out = os.path.join(os.path.dirname(IMG_DIR), "exp8_yolov3_" + os.path.basename(p))
        cv2.imwrite(out, result)
        print("已保存:", out)


if __name__ == "__main__":
    main()
