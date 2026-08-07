# -*- coding: utf-8 -*-
"""
utils.py —— 公共工具函数: 数据加载 / 边划分 / 评估 / t-SNE 可视化 / 结果保存
"""
import os
import numpy as np
import torch
import dgl
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目根目录 / 结果目录 / 数据目录
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'result')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
# 让 DGL 数据集下载到项目的 data 目录(而不是用户主目录)
os.environ.setdefault('DGL_DOWNLOAD_DIR', os.path.join(DATA_DIR, 'dgl'))

# 本机为虚拟机, 小张量操作在线程数过高时同步开销反而更大; 限制为 min(8, cpu)
import torch as _torch
_torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_csv(rows, name):
    """rows: list[dict] -> 保存为 csv"""
    import csv
    path = os.path.join(RESULTS_DIR, name)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'[结果已保存] {path}')
    return path


def load_node_dataset(name='cora'):
    """加载 DGL 节点级数据集, 返回 (graph, features, labels, num_classes, num_nodes)
    注意: 保持原始图结构(不加自环), 需要自环的模型在各自实验脚本中自行添加."""
    datasets = {
        'cora': dgl.data.CoraGraphDataset,
        'citeseer': dgl.data.CiteseerGraphDataset,
        'pubmed': dgl.data.PubmedGraphDataset,
    }
    ds = datasets[name]()
    g = ds[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    num_classes = ds.num_classes
    num_nodes = g.num_nodes()
    return g, features, labels, num_classes, num_nodes


def split_edges(g, train_ratio=0.9, seed=42):
    """
    将图的边划分为训练/测试集合.
    返回: train_g(去掉了测试正边的图), test_pos(测试正边), test_neg(测试负边),
          train_pos(训练正边), train_neg(训练负边)
    """
    set_seed(seed)
    u, v = g.edges()
    edges = np.stack([u.numpy(), v.numpy()], axis=1)
    n = g.num_nodes()
    n_test = int(round(len(edges) * (1 - train_ratio)))

    # 挑测试正边(确保没有自环)
    test_idx = np.random.choice(len(edges), n_test, replace=False)
    test_mask = np.zeros(len(edges), dtype=bool)
    test_mask[test_idx] = True
    test_pos = edges[test_mask]
    train_pos = edges[~test_mask]

    # 训练图上移除测试正边
    train_g = dgl.remove_edges(g, torch.from_numpy(test_idx).long())
    train_g = dgl.add_self_loop(train_g)

    # 构造负边(不存在的节点对)
    test_neg = sample_negative_edges(n, test_pos)
    train_neg = sample_negative_edges(n, train_pos)
    return train_g, test_pos, test_neg, train_pos, train_neg


def sample_negative_edges(n, pos_edges, seed=None):
    """随机采样 n_test 条不存在的边"""
    rng = np.random.RandomState(seed)
    pos_set = set(map(tuple, pos_edges.tolist()))
    negs = []
    while len(negs) < len(pos_edges):
        a = rng.randint(0, n, len(pos_edges) * 2)
        b = rng.randint(0, n, len(pos_edges) * 2)
        cand = np.stack([a, b], axis=1)
        for e in cand:
            if e[0] == e[1]:
                continue
            if (e[0], e[1]) in pos_set or (e[1], e[0]) in pos_set:
                continue
            negs.append(e)
            if len(negs) >= len(pos_edges):
                break
    return np.array(negs)[:len(pos_edges)]


def evaluate_link_prediction(emb, pos_edges, neg_edges):
    """
    基于 embedding 计算边得分并评测 AUC.
    emb: [N, D]; pos_edges/neg_edges: [M, 2]
    """
    pos_src = pos_edges[:, 0]
    pos_dst = pos_edges[:, 1]
    neg_src = neg_edges[:, 0]
    neg_dst = neg_edges[:, 1]
    pos_score = np.sum(emb[pos_src] * emb[pos_dst], axis=1)
    neg_score = np.sum(emb[neg_src] * emb[neg_dst], axis=1)
    y = np.concatenate([np.ones(len(pos_score)), np.zeros(len(neg_score))])
    score = np.concatenate([pos_score, neg_score])
    return roc_auc_score(y, score)


def tsne_plot(emb, labels, title, path, perplexity=30, n_sample=None):
    """对 embedding 做 t-SNE 降维并保存散点图"""
    labels = np.asarray(labels)
    if n_sample is not None and len(emb) > n_sample:
        idx = np.random.RandomState(0).choice(len(emb), n_sample, replace=False)
        emb, labels = emb[idx], labels[idx]
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=1000, random_state=42)
    emb2 = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(emb2[:, 0], emb2[:, 1], c=labels, cmap='tab10',
                         s=8, alpha=0.7)
    cbar = fig.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 维度 1')
    ax.set_ylabel('t-SNE 维度 2')
    fig.tight_layout()
    full = os.path.join(RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')
    return full


def plot_training_curves(losses_dict, title, path):
    """绘制训练 loss 曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, losses in losses_dict.items():
        ax.plot(losses, marker='o', label=name)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    full = os.path.join(RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')
    return full


def plot_grouped_bar(labels, data_dict, title, path, ylabel='指标值', rot=0):
    """分组柱状图: data_dict[name] -> 每组各柱的值"""
    x = np.arange(len(labels))
    width = 0.8 / len(data_dict)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, (name, vals) in enumerate(data_dict.items()):
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width * (len(data_dict) - 1) / 2)
    ax.set_xticklabels(labels, rotation=rot)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    full = os.path.join(RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')
    return full
