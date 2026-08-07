# -*- coding: utf-8 -*-
"""
dataset_viz.py —— 数据集概览与可视化

下载并展示 Cora / Citeseer / Pubmed 三个数据集:
  - 基本信息统计表 (节点数/边数/类别数/特征维/标签分布/划分大小)
  - 标签分布柱状图
  - 节点度分布图 (log-log)
  - 节点特征 t-SNE 降维(按标签着色)
  - 随机采样子图的网络拓扑布局
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dgl
import networkx as nx
from collections import Counter

import utils

DATASETS = ['cora', 'citeseer', 'pubmed']
N_SAMPLE_TSNE = 2000
SUBGRAPH_NODES = 250


def load(name):
    """加载数据集, 返回 (g, features, labels, num_classes)"""
    return utils.load_node_dataset(name)


def subgraph_sample(g, n_nodes=250, seed=0):
    """从图中随机采样一个连通的诱导子图用于拓扑可视化"""
    rng = np.random.RandomState(seed)
    # DGL 内置的 Cora/Citeseer/Pubmed 数据集已是双向(无向)存储, 只需加自环便于 BFS
    g_und = dgl.add_self_loop(g)
    # 随机起点 BFS
    start = int(rng.randint(0, g.num_nodes()))
    visited, frontier = {start}, [start]
    while len(visited) < n_nodes and frontier:
        nxt = []
        for u in frontier:
            succ = g_und.successors(u).tolist()
            rng.shuffle(succ)
            for v in succ:
                if v not in visited and len(visited) < n_nodes:
                    visited.add(v)
                    nxt.append(v)
        frontier = nxt
    idx = sorted(visited)
    sg = dgl.node_subgraph(g, idx)
    return sg


def plot_label_distribution(labels, num_classes, title, path):
    counts = Counter(labels.tolist())
    cats = [counts.get(i, 0) for i in range(num_classes)]
    utils.plot_grouped_bar([str(i) for i in range(num_classes)],
                           {'节点数': cats}, title, path, ylabel='节点数')


def plot_degree_distribution(g, title, path):
    deg = g.in_degrees().numpy()
    hist, edges = np.histogram(deg, bins=np.logspace(np.log10(max(deg.min(),1)),
                                                     np.log10(deg.max()), 30))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(edges[:-1], hist, 'o-', alpha=0.8)
    ax.set_xlabel('节点度 d')
    ax.set_ylabel('节点数 (log)')
    ax.set_title(title)
    fig.tight_layout()
    full = os.path.join(utils.RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')


def plot_feature_tsne(features, labels, title, path):
    labels = labels.numpy()
    if len(features) > N_SAMPLE_TSNE:
        idx = np.random.RandomState(0).choice(len(features), N_SAMPLE_TSNE, replace=False)
        feats, labels = features[idx], labels[idx]
    utils.tsne_plot(feats.numpy(), labels, title, path, n_sample=None)


def plot_subgraph_topology(g, labels, title, path):
    sg = subgraph_sample(g, SUBGRAPH_NODES)
    sub_labels = labels[sg.ndata[dgl.NID].numpy()]
    G = dgl.to_networkx(sg).to_undirected()
    pos = nx.spring_layout(G, seed=42, k=0.7)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(sub_labels / max(sub_labels.max(), 1))
    nx.draw_networkx(G, pos, node_color=colors, node_size=40, with_labels=False,
                     width=0.3, edge_color='lightgray', ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    full = os.path.join(utils.RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')


def main():
    stats_rows = []
    for ds_name in DATASETS:
        print(f'===== 加载并可视化数据集: {ds_name} =====')
        g, features, labels, num_classes, num_nodes = load(ds_name)
        deg = g.in_degrees().numpy()
        label_counts = Counter(labels.numpy().tolist())
        n_train = int(g.ndata['train_mask'].sum())
        n_val = int(g.ndata['val_mask'].sum())
        n_test = int(g.ndata['test_mask'].sum())
        stats_rows.append({
            '数据集': ds_name, '节点数': num_nodes, '边数(有向)': g.num_edges(),
            '特征维数': features.shape[1], '类别数': num_classes,
            '平均度': round(deg.mean(), 2), '训练节点': n_train,
            '验证节点': n_val, '测试节点': n_test,
            '每类节点数': '; '.join(f'{k}:{v}' for k, v in sorted(label_counts.items()))
        })

        # 标签分布
        plot_label_distribution(labels.numpy(), num_classes,
                                f'{ds_name} 标签分布', f'dataset_{ds_name}_labels.png')
        # 度分布
        plot_degree_distribution(g, f'{ds_name} 节点度分布', f'dataset_{ds_name}_degree.png')
        # 特征 t-SNE
        plot_feature_tsne(features, labels, f'{ds_name} 节点特征 t-SNE (按标签着色)',
                          f'dataset_{ds_name}_feature_tsne.png')
        # 子图拓扑
        plot_subgraph_topology(g, labels, f'{ds_name} 子图拓扑(按标签着色)',
                               f'dataset_{ds_name}_topology.png')

    utils.save_csv(stats_rows, 'dataset_统计对比.csv')
    print('\n===== 三数据集统计对比 =====')
    for r in stats_rows:
        print(f"  {r['数据集']:9s} 节点 {r['节点数']:6d}  边 {r['边数(有向)']:7d}  "
              f"特征 {r['特征维数']:5d}  类 {r['类别数']}  训练/验证/测试 "
              f"{r['训练节点']}/{r['验证节点']}/{r['测试节点']}")
    print('数据集可视化完成 ✔')


if __name__ == '__main__':
    main()
