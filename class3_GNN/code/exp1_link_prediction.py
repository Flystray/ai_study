# -*- coding: utf-8 -*-
"""
exp1_link_prediction.py —— 实验①  Node embedding · Link prediction

任务要点:
  - 数据集: Cora(可换 citeseer / pubmed), 使用 dgl.data.CoraGraphDataset()
  - 将图的边划分为训练/测试集合
  - 基于 dgl.sampling.random_walk / node2vec_random_walk 训练
    DeepWalk / node2vec / LINE 得到节点 embedding
  - 在测试集与训练集上评测链接预测 AUC
  - 超参实验: DeepWalk(路径长度, 每节点路径数) / node2vec(p,q) / LINE(一/二阶, 负样本数)
  - 可视化: 对三种模型的 embedding 做 t-SNE 降维
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dgl

import utils
import models
from models import train_skipgram, train_line

EMB_DIM = 128

# ---------------------------------------------------------------------------
# 随机游走生成
# ---------------------------------------------------------------------------
def make_walks_deepwalk(g, num_nodes, num_walks, walk_length, seed=0):
    """基于 dgl.sampling.random_walk 生成 DeepWalk 随机游走序列"""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    walks = []
    batch = 512
    for start in range(0, num_nodes, batch):
        nodes = torch.arange(start, min(start + batch, num_nodes))
        for _ in range(num_walks):
            traces, _ = dgl.sampling.random_walk(g, nodes, length=walk_length)
            walks.append(traces.numpy())
    return walks


def make_walks_node2vec(g, num_nodes, p, q, num_walks, walk_length, seed=0):
    """基于 dgl.sampling.node2vec_random_walk 生成 node2vec 随机游走序列
    (注意: 该接口返回单个 Tensor, 形状 [B, walk_length+1])"""
    torch.manual_seed(seed)
    walks = []
    batch = 512
    for start in range(0, num_nodes, batch):
        nodes = torch.arange(start, min(start + batch, num_nodes))
        for _ in range(num_walks):
            traces = dgl.sampling.node2vec_random_walk(g, nodes, p=p, q=q,
                                                       walk_length=walk_length)
            walks.append(traces.numpy())
    return walks


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(dataset='cora', seed=42):
    utils.set_seed(seed)
    print(f'===== 实验① Node embedding - Link prediction (数据集: {dataset}) =====')

    g, features, labels, num_classes, num_nodes = utils.load_node_dataset(dataset)
    print(f'节点数 {g.num_nodes()}, 边数(原始) {g.num_edges()}, 类别数 {num_classes}')

    # 1) 边划分: 90% 训练边 / 10% 测试边, 并构造负样本边
    train_g, test_pos, test_neg, train_pos, train_neg = utils.split_edges(
        g, train_ratio=0.9, seed=seed)
    print(f'训练正边 {len(train_pos)}, 测试正边 {len(test_pos)}, 测试负边 {len(test_neg)}')

    # 2) 主实验: 三种模型在默认超参下训练并评测
    default_results = {}
    # --- DeepWalk ---
    walks_dw = make_walks_deepwalk(train_g, num_nodes, num_walks=10,
                                   walk_length=40, seed=seed)
    emb_dw, _ = train_skipgram(walks_dw, num_nodes, emb_dim=EMB_DIM, epochs=2)
    auc_test_dw = utils.evaluate_link_prediction(emb_dw, test_pos, test_neg)
    auc_train_dw = utils.evaluate_link_prediction(emb_dw, train_pos, train_neg)
    default_results['DeepWalk'] = (auc_train_dw, auc_test_dw, emb_dw)

    # --- node2vec ---
    walks_n2v = make_walks_node2vec(train_g, num_nodes, p=1.0, q=1.0,
                                    num_walks=10, walk_length=40, seed=seed)
    emb_n2v, _ = train_skipgram(walks_n2v, num_nodes, emb_dim=EMB_DIM, epochs=2)
    auc_test_n2v = utils.evaluate_link_prediction(emb_n2v, test_pos, test_neg)
    auc_train_n2v = utils.evaluate_link_prediction(emb_n2v, train_pos, train_neg)
    default_results['node2vec'] = (auc_train_n2v, auc_test_n2v, emb_n2v)

    # --- LINE(默认二阶) ---
    emb_line, _ = train_line(train_g, num_nodes, emb_dim=EMB_DIM, order='second',
                             negative=5, epochs=10)
    auc_test_line = utils.evaluate_link_prediction(emb_line, test_pos, test_neg)
    auc_train_line = utils.evaluate_link_prediction(emb_line, train_pos, train_neg)
    default_results['LINE'] = (auc_train_line, auc_test_line, emb_line)

    # 主实验结果表
    main_rows = [{
        '模型': k, '训练集AUC': round(v[0], 4), '测试集AUC': round(v[1], 4),
        '差值(训练-测试)': round(v[0] - v[1], 4)
    } for k, v in default_results.items()]
    utils.save_csv(main_rows, 'exp1_主实验_link_prediction.csv')
    for r in main_rows:
        print(f"  {r['模型']:10s} 训练AUC {r['训练集AUC']:.4f}  测试AUC {r['测试集AUC']:.4f}")

    # 3) t-SNE 可视化
    labels_arr = labels.numpy()
    for name in ['DeepWalk', 'node2vec', 'LINE']:
        emb = default_results[name][2]
        utils.tsne_plot(emb, labels_arr, f'{name} Node Embedding (t-SNE)',
                        f'exp1_tsne_{name}.png', n_sample=1500)

    # 4) 超参实验
    hyper_rows = []

    # 4.1 DeepWalk: 路径长度
    for wl in [20, 40, 80]:
        wk = make_walks_deepwalk(train_g, num_nodes, num_walks=10,
                                 walk_length=wl, seed=seed)
        emb, _ = train_skipgram(wk, num_nodes, emb_dim=EMB_DIM, epochs=1)
        auc = utils.evaluate_link_prediction(emb, test_pos, test_neg)
        hyper_rows.append({'模型': 'DeepWalk', '超参': '路径长度', '值': wl, '测试AUC': round(auc, 4)})
    # 4.1 DeepWalk: 每节点起始路径数
    for nw in [5, 10, 20]:
        wk = make_walks_deepwalk(train_g, num_nodes, num_walks=nw,
                                 walk_length=40, seed=seed)
        emb, _ = train_skipgram(wk, num_nodes, emb_dim=EMB_DIM, epochs=1)
        auc = utils.evaluate_link_prediction(emb, test_pos, test_neg)
        hyper_rows.append({'模型': 'DeepWalk', '超参': '每节点路径数', '值': nw, '测试AUC': round(auc, 4)})

    # 4.2 node2vec: p,q
    for p, q in [(0.25, 0.25), (0.25, 4.0), (1.0, 1.0), (4.0, 0.25), (4.0, 4.0)]:
        wk = make_walks_node2vec(train_g, num_nodes, p=p, q=q, num_walks=10,
                                 walk_length=40, seed=seed)
        emb, _ = train_skipgram(wk, num_nodes, emb_dim=EMB_DIM, epochs=1)
        auc = utils.evaluate_link_prediction(emb, test_pos, test_neg)
        hyper_rows.append({'模型': 'node2vec', '超参': f'p,q=({p},{q})', '值': '-',
                           '测试AUC': round(auc, 4)})

    # 4.3 LINE: 两种损失(一阶/二阶)
    for order in ['first', 'second']:
        emb, _ = train_line(train_g, num_nodes, emb_dim=EMB_DIM, order=order,
                            negative=5, epochs=10)
        auc = utils.evaluate_link_prediction(emb, test_pos, test_neg)
        hyper_rows.append({'模型': 'LINE', '超参': '损失函数(阶数)', '值': order,
                           '测试AUC': round(auc, 4)})
    # 4.3 LINE: 负样本数量
    for neg in [1, 5, 10]:
        emb, _ = train_line(train_g, num_nodes, emb_dim=EMB_DIM, order='second',
                            negative=neg, epochs=10)
        auc = utils.evaluate_link_prediction(emb, test_pos, test_neg)
        hyper_rows.append({'模型': 'LINE', '超参': '负样本数', '值': neg,
                           '测试AUC': round(auc, 4)})

    utils.save_csv(hyper_rows, 'exp1_超参实验_link_prediction.csv')
    for r in hyper_rows:
        print(f"  {r['模型']:9s} {r['超参']:14s} {str(r['值']):8s} 测试AUC {r['测试AUC']:.4f}")

    print('实验① 完成 ✔')


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'cora'
    run(ds)
