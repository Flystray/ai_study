# -*- coding: utf-8 -*-
"""
把下载好的 ImageNet 验证集组织成 torchvision ImageFolder 结构：
    data/imagenet/val/<synset>/ILSVRC2012_val_xxxxx.JPEG
标签来源：ILSVRC2012_validation_ground_truth.txt（50000 行，类别索引 1-1000）
synset 映射来源：devkit meta.mat（已存 synset_map.json）
"""
import os, json, tarfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

TAR = os.path.join(DATA, "imagenet_val", "ILSVRC2012_img_val.tar")
GT = os.path.join(DATA, "imagenet_devkit", "ILSVRC2012_devkit_t12", "data",
                  "ILSVRC2012_validation_ground_truth.txt")
MAP = os.path.join(DATA, "imagenet_devkit", "synset_map.json")
RAW = os.path.join(DATA, "imagenet_val_raw")      # 解压的扁平图片
VAL = os.path.join(DATA, "imagenet", "val")       # 目标 ImageFolder 结构


def main():
    if not os.path.exists(TAR):
        print("验证集 tar 不存在:", TAR); return
    with open(GT) as f:
        labels = [int(x.strip()) for x in f.readlines()]
    with open(MAP) as f:
        synset_map = {int(k): v for k, v in json.load(f).items()}
    print(f"标签数: {len(labels)} | synset 映射类数: {len(synset_map)}")

    # 1. 解压 tar（若未解压）
    if not os.path.isdir(RAW) or len(os.listdir(RAW)) < 1000:
        print("解压验证集到", RAW, "...")
        os.makedirs(RAW, exist_ok=True)
        # 清空旧内容
        for f in os.listdir(RAW):
            os.remove(os.path.join(RAW, f))
        with tarfile.open(TAR) as tar:
            tar.extractall(path=RAW)
        print("解压完成，文件数:", len(os.listdir(RAW)))
    else:
        print("已存在解压目录:", RAW)

    # 2. 按标签组织到 val/<synset>/
    files = sorted(os.listdir(RAW))
    assert len(files) == len(labels), f"文件数 {len(files)} 与标签数 {len(labels)} 不一致"
    print("组织目录结构（每个类别一个 synset 子目录）...")
    os.makedirs(VAL, exist_ok=True)
    moved = 0
    for fname, idx in zip(files, labels):
        wnid, words = synset_map.get(idx, (f"unknown_{idx}", "unknown"))
        d = os.path.join(VAL, wnid)
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, fname)
        if not os.path.exists(dst):
            shutil.move(os.path.join(RAW, fname), dst)
        moved += 1
    # 清理空目录
    for root, dirs, fnames in os.walk(RAW):
        for d in dirs:
            p = os.path.join(root, d)
            if not os.listdir(p):
                os.rmdir(p)
    n_class = len(os.listdir(VAL))
    print(f"完成：{moved} 张图片已组织到 {VAL}，共 {n_class} 个 synset 类别目录")


if __name__ == "__main__":
    main()
