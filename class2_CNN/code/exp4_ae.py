# -*- coding: utf-8 -*-
"""
实验 4：自编码器（AutoEncoder）—— 基于 CNN 的 AE vs 基于全连接的 AE
数据集：MNIST-digits / MNIST-Fashion
指标：重建 MSE、重构效果可视化对比。
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
from common import (set_seed, DEVICE, load_dataset, RESULTS_DIR, plot_image_grid)

set_seed(42)

EPOCHS = 1 if os.environ.get("SMOKE") else 15
BATCH, LR = 128, 1e-3
DATASETS = ["mnist", "fashion"]


class ConvAE(nn.Module):
    """基于 CNN 的 AE：卷积下采样编码 → 潜在向量 → 反卷积上采样解码。"""
    def __init__(self, in_channels=1, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),   # 14x14
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(True),   # 7x7
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(True),  # 4x4
        )
        self.fc_in = 128 * 4 * 4
        self.fc = nn.Linear(self.fc_in, latent)
        self.fc_back = nn.Linear(latent, self.fc_in)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(True),  # 8x8
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(True),   # 16x16
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),         # 32x32 -> crop
        )
        self.crop = nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False)
        self.out = nn.Sigmoid()

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.fc(h)
        d = self.fc_back(z).view(-1, 128, 4, 4)
        return self.out(self.crop(self.decoder(d)))


class MLPAE(nn.Module):
    """基于全连接的 AE：784 → 256 → 64 → 256 → 784。"""
    def __init__(self, in_features=784, hidden=256, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(True),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(True),
            nn.Linear(hidden, in_features), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x.view(x.size(0), -1)))


def train_ae(model, train_loader, epochs, label=""):
    opt = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    losses = []
    for ep in range(1, epochs + 1):
        tl = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            rec = model(x)
            target = x.view(x.size(0), -1) if rec.dim() == 2 else x  # 兼容展平输出的 MLP-AE
            loss = criterion(rec, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * x.size(0)
        tl /= len(train_loader.dataset)
        losses.append(tl)
        if ep % 3 == 0 or ep == 1:
            print(f"[{label}] epoch {ep}/{epochs} | recon_loss={tl:.4f}")
    return losses


def main():
    summary = {}
    for dname in DATASETS:
        print("=" * 60)
        print(f"数据集: {dname}")
        # AE 用 [0,1] 输入（与 sigmoid 输出同量纲）
        train_loader, test_loader, in_ch, h, w, _, _ = load_dataset(dname, batch_size=BATCH, normalize=False)

        cnn_ae = ConvAE(in_channels=in_ch).to(DEVICE)
        mlp_ae = MLPAE(in_features=in_ch * h * w).to(DEVICE)

        print("[CNN-AE] 训练 ...")
        l_cnn = train_ae(cnn_ae, train_loader, EPOCHS, label=f"{dname}/cnn-ae")
        print("[MLP-AE] 训练 ...")
        l_mlp = train_ae(mlp_ae, train_loader, EPOCHS, label=f"{dname}/mlp-ae")

        # 测试集重建
        cnn_ae.eval(); mlp_ae.eval()
        xs = []
        with torch.no_grad():
            x, _ = next(iter(test_loader))
            x = x[:10].to(DEVICE)
            rec_cnn = cnn_ae(x).cpu().numpy()
            rec_mlp = mlp_ae(x).cpu().numpy()
            if rec_mlp.ndim == 2:  # MLP-AE 展平输出 -> 还原为图像
                rec_mlp = rec_mlp.reshape(-1, in_ch, h, w)
            x_np = x.cpu().numpy()

        # 可视化：原图 / CNN 重建 / MLP 重建
        fig, axes = plt.subplots(3, 10, figsize=(15, 4.5))
        for j in range(10):
            for row, img in enumerate([x_np, rec_cnn, rec_mlp]):
                axes[row, j].imshow(img[j].squeeze(), cmap="gray")
                axes[row, j].axis("off")
        axes[0, 0].set_ylabel("原图", fontsize=10)
        axes[1, 0].set_ylabel("CNN-AE", fontsize=10)
        axes[2, 0].set_ylabel("MLP-AE", fontsize=10)
        fig.suptitle(f"{dname}: 自编码器重构对比", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"exp4_recon_{dname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 训练损失曲线
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, EPOCHS + 1), l_cnn, marker="o", label="CNN-AE")
        ax.plot(range(1, EPOCHS + 1), l_mlp, marker="s", label="MLP-AE")
        ax.set_xlabel("Epoch"); ax.set_ylabel("重建 MSE"); ax.set_title(f"{dname}: AE 训练损失")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f"exp4_loss_{dname}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        summary[dname] = {"cnn_ae_final_loss": l_cnn[-1], "mlp_ae_final_loss": l_mlp[-1]}

    with open(os.path.join(RESULTS_DIR, "exp4_ae.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("实验 4 完成，结果见 results/exp4_*")


if __name__ == "__main__":
    main()
