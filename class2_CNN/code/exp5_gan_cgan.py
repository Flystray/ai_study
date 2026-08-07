# -*- coding: utf-8 -*-
"""
实验 5：生成式对抗网络 GAN 与条件 GAN（CGAN），数据集 MNIST-digits。
目标：
  - GAN：学习数据分布，生成逼真的手写数字；
  - CGAN：给定标签条件，生成指定数字；
对比普通生成与条件生成的效果。
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (set_seed, DEVICE, load_dataset, RESULTS_DIR)

set_seed(42)

EPOCHS = 1 if os.environ.get("SMOKE") else 15
BATCH, Z_DIM = 128, 100
LR, BETA = 2e-4, (0.5, 0.999)


class Generator(nn.Module):
    """z -> 28x28 图像"""
    def __init__(self, z_dim=Z_DIM, n_classes=None):
        super().__init__()
        c_in = z_dim + (10 if n_classes else 0)
        self.fc = nn.Linear(c_in, 128 * 7 * 7)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1), nn.Tanh(),
        )

    def forward(self, z, labels=None):
        if labels is not None:
            z = torch.cat([z, labels], dim=1)
        return self.main(self.fc(z).view(z.size(0), 128, 7, 7))


class Discriminator(nn.Module):
    """28x28 图像 -> 真/假"""
    def __init__(self, n_classes=None):
        super().__init__()
        c_in = 1 + (1 if n_classes else 0)
        self.embed_fc = nn.Linear(10, 1 * 28 * 28) if n_classes else None
        self.main = nn.Sequential(
            nn.Conv2d(c_in, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
        )
        self.out = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x, labels=None):
        if labels is not None:
            y = self.embed_fc(labels).view(labels.size(0), 1, 28, 28)
            x = torch.cat([x, y], dim=1)
        return self.out(self.main(x).view(x.size(0), -1))


def train_gan(dname, conditional=False, epochs=EPOCHS):
    print("=" * 60)
    tag = "CGAN" if conditional else "GAN"
    print(f"[{tag}] 训练 ...")
    train_loader, test_loader, in_ch, h, w, ncls, _ = load_dataset(dname, batch_size=BATCH)
    n_classes = 10 if conditional else None

    G = Generator(Z_DIM, n_classes).to(DEVICE)
    D = Discriminator(n_classes).to(DEVICE)
    optG = optim.Adam(G.parameters(), lr=LR, betas=BETA)
    optD = optim.Adam(D.parameters(), lr=LR, betas=BETA)
    bce = nn.BCEWithLogitsLoss()

    d_losses, g_losses = [], []
    fixed_z = torch.randn(100, Z_DIM, device=DEVICE)
    fixed_labels = torch.arange(10, device=DEVICE).repeat_interleave(10)  # 每标签10张

    for ep in range(1, epochs + 1):
        for i, (x, y) in enumerate(train_loader):
            real = x.to(DEVICE)
            real_y = torch.ones(real.size(0), 1, device=DEVICE)
            fake_y = torch.zeros(real.size(0), 1, device=DEVICE)
            z = torch.randn(real.size(0), Z_DIM, device=DEVICE)
            labels = None
            if conditional:
                labels = nn.functional.one_hot(y.to(DEVICE), 10).float()
                real_y = real_y * 0.9          # label smoothing
                labels_real = labels

            # ---- 判别器 ----
            optD.zero_grad()
            fake = G(z, labels)
            d_real = D(real, labels_real if conditional else None)
            d_fake = D(fake.detach(), labels if conditional else None)
            loss_d = bce(d_real, real_y) + bce(d_fake, fake_y)
            loss_d.backward(); optD.step()

            # ---- 生成器 ----
            optG.zero_grad()
            z = torch.randn(real.size(0), Z_DIM, device=DEVICE)
            fake = G(z, labels)
            g_loss = bce(D(fake, labels if conditional else None), torch.ones_like(fake_y))
            g_loss.backward(); optG.step()

        d_losses.append(loss_d.item()); g_losses.append(g_loss.item())

        # 每 5 epoch 存一次生成样本
        G.eval()
        if ep % 5 == 0 or ep == epochs:
            with torch.no_grad():
                if conditional:
                    samples = G(fixed_z, nn.functional.one_hot(fixed_labels, 10).float())
                else:
                    samples = G(fixed_z)
                samples = samples.cpu().numpy()
            plot_gen(samples, os.path.join(RESULTS_DIR, f"exp5_{tag.lower()}_{dname}_ep{ep}.png"),
                     nrow=10, title=f"{tag} epoch {ep}")
        G.train()
        print(f"[{tag}] epoch {ep}/{epochs} | D_loss={d_losses[-1]:.4f} G_loss={g_losses[-1]:.4f}")

    # 最终生成样本 + 训练曲线
    G.eval()
    with torch.no_grad():
        if conditional:
            samples = G(fixed_z, nn.functional.one_hot(fixed_labels, 10).float())
        else:
            samples = G(fixed_z)
        samples = samples.cpu().numpy()
    plot_gen(samples, os.path.join(RESULTS_DIR, f"exp5_{tag.lower()}_{dname}_final.png"),
             nrow=10, title=f"{tag} 最终生成结果（每列/行一组标签）" if conditional else f"{tag} 最终生成结果")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(d_losses, label="D loss"); ax.plot(g_losses, label="G loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(f"{tag} 训练曲线")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, f"exp5_{tag.lower()}_{dname}_loss.png"), dpi=150)
    plt.close(fig)
    return {"d_loss": d_losses[-1], "g_loss": g_losses[-1]}


def plot_gen(samples, path, nrow=10, title=""):
    """samples: (N,1,28,28) 范围 [-1,1]"""
    s = (np.clip(samples, -1, 1) + 1) / 2   # 映射到 [0,1]
    import torchvision
    t = torch.from_numpy(s[:100])
    grid = torchvision.utils.make_grid(t, nrow=nrow, padding=1)
    fig, ax = plt.subplots(figsize=(nrow * 0.9, (100 / nrow) * 0.9))
    ax.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    ax.axis("off"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("已保存:", path)


if __name__ == "__main__":
    summary = {
        "gan": train_gan("mnist", conditional=False),
        "cgan": train_gan("mnist", conditional=True),
    }
    with open(os.path.join(RESULTS_DIR, "exp5_gan_cgan.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("实验 5 完成，结果见 results/exp5_*")
