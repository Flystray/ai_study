# -*- coding: utf-8 -*-
"""
选做实验 7：Pix2Pix 图像风格迁移（Image-to-Image Translation）
论文：Image-to-Image Translation with Conditional Adversarial Networks (Isola et al., 2017)
方法：条件 GAN —— 生成器（UNet）+ 判别器（PatchGAN），
      损失 = 对抗损失(BCE) + L1 重建损失 × lambda。
数据集：pix2pix 官方 facades 数据集（边缘图 -> 建筑立面图）。
运行：首次自动下载数据集（约 70MB）。
注意：本实验为选做，按需运行；CPU 训练较慢，建议减小 EPOCHS。
"""
import os, sys, time, tarfile, urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEVICE, RESULTS_DIR, PROJECT_ROOT

DATA_URL = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "pix2pix", "facades")  # 解压后的子目录
EPOCHS = 1 if os.environ.get("SMOKE") else 8
LAMBDA_L1 = 100
BATCH = 8
IMG_SIZE = 256


# ---------- 数据 ----------
def download_and_prepare():
    parent = os.path.dirname(DATA_DIR)          # data/pix2pix（tar 解压出 facades/ 子目录）
    os.makedirs(parent, exist_ok=True)
    if not os.path.isdir(os.path.join(DATA_DIR, "train")):
        print("下载 facades 数据集 ...")
        tarball = os.path.join(parent, "facades.tar.gz")
        urllib.request.urlretrieve(DATA_URL, tarball)
        with tarfile.open(tarball) as tar:
            tar.extractall(path=parent)
        os.remove(tarball)
        print("数据就绪:", DATA_DIR)


class FacadesDataset(Dataset):
    """每张图左半为输入（边缘图），右半为真实图，训练时在通道维拼接。"""
    def __init__(self, root, split="train", size=IMG_SIZE):
        self.files = sorted(os.listdir(os.path.join(root, split)))
        self.root = root; self.split = split; self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.split, self.files[i])).convert("RGB")
        w, h = img.size
        img = img.resize((self.size * 2, self.size))
        arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
        left = torch.from_numpy(arr[:, :self.size].transpose(2, 0, 1))
        right = torch.from_numpy(arr[:, self.size:].transpose(2, 0, 1))
        return left, right


# ---------- 网络 ----------
class UnetGenerator(nn.Module):
    """UNet：编码-解码 + 跳连，输入输出同尺寸。"""
    def __init__(self, in_ch=3, out_ch=3, nf=64):
        super().__init__()
        def conv(in_, out, norm=True):
            l = [nn.Conv2d(in_, out, 4, 2, 1)]
            if norm: l.append(nn.BatchNorm2d(out))
            l.append(nn.LeakyReLU(0.2))
            return l
        def deconv(in_, out):
            return nn.Sequential(
                nn.ConvTranspose2d(in_, out, 4, 2, 1), nn.BatchNorm2d(out), nn.ReLU(True))
        self.e1 = nn.Sequential(*conv(in_ch, nf, norm=False))
        self.e2 = nn.Sequential(*conv(nf, nf*2))
        self.e3 = nn.Sequential(*conv(nf*2, nf*4))
        self.e4 = nn.Sequential(*conv(nf*4, nf*8))
        self.e5 = nn.Sequential(*conv(nf*8, nf*8))
        self.b = nn.Sequential(nn.Conv2d(nf*8, nf*8, 4, 2, 1), nn.ReLU(True))
        self.d5 = deconv(nf*8, nf*8); self.d4 = deconv(nf*8*2, nf*8)
        self.d3 = deconv(nf*8*2, nf*4); self.d2 = deconv(nf*4*2, nf*2)
        self.d1 = deconv(nf*2*2, nf)
        self.out = nn.Sequential(nn.ConvTranspose2d(nf*2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        e4 = self.e4(e3); e5 = self.e5(e4)
        b = self.b(e5)
        d5 = self.d5(b); d4 = self.d4(torch.cat([d5, e5], 1))
        d3 = self.d3(torch.cat([d4, e4], 1)); d2 = self.d2(torch.cat([d3, e3], 1))
        d1 = self.d1(torch.cat([d2, e2], 1))
        return self.out(torch.cat([d1, e1], 1))


class PatchGAN(nn.Module):
    """70x70 PatchGAN 判别器。"""
    def __init__(self, in_ch=6, nf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, nf, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(nf, nf*2, 4, 2, 1), nn.BatchNorm2d(nf*2), nn.LeakyReLU(0.2),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1), nn.BatchNorm2d(nf*4), nn.LeakyReLU(0.2),
            nn.Conv2d(nf*4, nf*8, 4, 1, 1), nn.BatchNorm2d(nf*8), nn.LeakyReLU(0.2),
            nn.Conv2d(nf*8, 1, 4, 1, 1),
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))


# ---------- 训练 ----------
def train():
    download_and_prepare()
    train_ds = FacadesDataset(DATA_DIR, "train")
    val_ds = FacadesDataset(DATA_DIR, "val")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    G = UnetGenerator().to(DEVICE)
    D = PatchGAN().to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    for ep in range(1, EPOCHS + 1):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 判别器输出尺寸 = PatchGAN 输出（随输入尺寸变化），动态生成标签
            d_real = D(x, y)
            real = torch.ones_like(d_real)   # PatchGAN 输出
            fake = torch.zeros_like(real)

            # 判别器
            optD.zero_grad()
            fake_y = G(x)
            d_loss = (bce(D(x, y), real) + bce(D(x, fake_y.detach()), fake)) / 2
            d_loss.backward(); optD.step()

            # 生成器
            optG.zero_grad()
            fake_y = G(x)
            g_adv = bce(D(x, fake_y), real)
            g_l1 = l1(fake_y, y) * LAMBDA_L1
            g_loss = g_adv + g_l1
            g_loss.backward(); optG.step()

        # 验证可视化
        if ep % 2 == 0 or ep == EPOCHS:
            G.eval()
            with torch.no_grad():
                xv, yv = next(iter(val_loader))
                xv, yv = xv[:3].to(DEVICE), yv[:3].to(DEVICE)
                out = G(xv).cpu().numpy()
            rows = []
            for j in range(3):
                row = np.concatenate([((xv[j].cpu().numpy().transpose(1,2,0)+1)/2),
                                      ((out[j].transpose(1,2,0)+1)/2),
                                      ((yv[j].cpu().numpy().transpose(1,2,0)+1)/2)], axis=1)
                rows.append(row)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(np.concatenate(rows, axis=0)); ax.axis("off")
            ax.set_title(f"Pix2Pix epoch {ep}：输入 | 生成 | 真实")
            fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"exp7_pix2pix_ep{ep}.png"),
                                            dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"[Pix2Pix] epoch {ep}/{EPOCHS} | D={d_loss.item():.4f} G={g_loss.item():.4f}")
            G.train()


if __name__ == "__main__":
    train()
    print("选做实验 7（Pix2Pix）完成，结果见 results/exp7_*")
