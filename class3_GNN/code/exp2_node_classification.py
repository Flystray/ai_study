# -*- coding: utf-8 -*-
"""
exp2_node_classification.py —— 实验②  Node embedding · Node classification

任务要点:
  - 数据集: Cora(可换 citeseer / pubmed)
  - 在整个图上训练得到节点 embedding (DeepWalk / node2vec / LINE)
  - 有标签节点划分训练/测试集, 用多分类器(Logistic / MLP)基于 embedding 分类
  - 评测指标 Accuracy
  - 超参实验: 与实验①类似的 embedding 超参 + 不同种类的多分类器
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import dgl

import utils
from exp1_link_prediction import make_walks_deepwalk, make_walks_node2vec
from models import train_skipgram, train_line
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

EMB_DIM = 128


def train_embedding(model_name, g, num_nodes, **kw):
    """训练指定模型得到节点 embedding"""
    if model_name == 'DeepWalk':
        wk = make_walks_deepwalk(g, num_nodes, num_walks=kw.get('num_walks', 10),
                                 walk_length=kw.get('walk_length', 40))
        emb, _ = train_skipgram(wk, num_nodes, emb_dim=EMB_DIM, epochs=2)
    elif model_name == 'node2vec':
        wk = make_walks_node2vec(g, num_nodes, p=kw.get('p', 1.0), q=kw.get('q', 1.0),
                                 num_walks=kw.get('num_walks', 10),
                                 walk_length=kw.get('walk_length', 40))
        emb, _ = train_skipgram(wk, num_nodes, emb_dim=EMB_DIM, epochs=2)
    else:  # LINE
        emb, _ = train_line(g, num_nodes, emb_dim=EMB_DIM,
                            order=kw.get('order', 'second'),
                            negative=kw.get('negative', 5), epochs=10)
    return emb


def build_classifier(name, hidden=None, seed=42):
    """构造多分类器: Logistic / MLP(1隐层) / MLP(2隐层)"""
    if name == 'Logistic':
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == 'MLP-1层':
        return MLPClassifier(hidden_layer_sizes=(64,), max_iter=500,
                             random_state=seed)
    if name == 'MLP-2层':
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             random_state=seed)
    raise ValueError(name)


def run(dataset='cora', seed=42):
    utils.set_seed(seed)
    print(f'===== 实验② Node embedding - Node classification (数据集: {dataset}) =====')

    g, features, labels, num_classes, num_nodes = utils.load_node_dataset(dataset)
    y = labels.numpy()
    print(f'节点数 {num_nodes}, 类别数 {num_classes}')

    # 有标签节点划分训练/测试集 (80/20)
    train_idx, test_idx = train_test_split(np.arange(num_nodes), test_size=0.2,
                                           random_state=seed, stratify=y)
    print(f'训练节点 {len(train_idx)}, 测试节点 {len(test_idx)}')

    # 1) 主实验: 三种 embedding 模型 × 默认 MLP-1层
    main_rows = []
    default_emb = {}
    for model_name in ['DeepWalk', 'node2vec', 'LINE']:
        emb = train_embedding(model_name, g, num_nodes)
        default_emb[model_name] = emb
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        train_acc = accuracy_score(y[train_idx], clf.predict(emb[train_idx]))
        test_acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        main_rows.append({'Embedding模型': model_name, '分类器': 'MLP-1层',
                          '训练集Acc': round(train_acc, 4),
                          '测试集Acc': round(test_acc, 4)})
        print(f"  {model_name:9s} 训练Acc {train_acc:.4f}  测试Acc {test_acc:.4f}")

    # 2) 超参实验 A: 不同种类的多分类器
    clf_rows = []
    for model_name, emb in default_emb.items():
        for clf_name in ['Logistic', 'MLP-1层', 'MLP-2层']:
            clf = build_classifier(clf_name)
            clf.fit(emb[train_idx], y[train_idx])
            test_acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
            clf_rows.append({'Embedding模型': model_name, '分类器': clf_name,
                             '测试集Acc': round(test_acc, 4)})
            print(f"  {model_name:9s} × {clf_name:8s} 测试Acc {test_acc:.4f}")

    # 3) 超参实验 B: embedding 超参 (固定 MLP-1层 分类器)
    hyp_rows = []
    # DeepWalk 路径长度 / 每节点路径数
    for wl in [20, 40, 80]:
        emb = train_embedding('DeepWalk', g, num_nodes, walk_length=wl)
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        hyp_rows.append({'Embedding模型': 'DeepWalk', '超参': '路径长度', '值': wl,
                         '测试Acc': round(acc, 4)})
    for nw in [5, 10, 20]:
        emb = train_embedding('DeepWalk', g, num_nodes, num_walks=nw)
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        hyp_rows.append({'Embedding模型': 'DeepWalk', '超参': '每节点路径数', '值': nw,
                         '测试Acc': round(acc, 4)})
    # node2vec p,q
    for p, q in [(0.25, 0.25), (1.0, 1.0), (4.0, 4.0)]:
        emb = train_embedding('node2vec', g, num_nodes, p=p, q=q)
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        hyp_rows.append({'Embedding模型': 'node2vec', '超参': f'p,q=({p},{q})',
                         '值': '-', '测试Acc': round(acc, 4)})
    # LINE 一/二阶 + 负样本数
    for order in ['first', 'second']:
        emb = train_embedding('LINE', g, num_nodes, order=order)
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        hyp_rows.append({'Embedding模型': 'LINE', '超参': '损失函数(阶数)', '值': order,
                         '测试Acc': round(acc, 4)})
    for neg in [1, 5, 10]:
        emb = train_embedding('LINE', g, num_nodes, negative=neg)
        clf = build_classifier('MLP-1层')
        clf.fit(emb[train_idx], y[train_idx])
        acc = accuracy_score(y[test_idx], clf.predict(emb[test_idx]))
        hyp_rows.append({'Embedding模型': 'LINE', '超参': '负样本数', '值': neg,
                         '测试Acc': round(acc, 4)})

    utils.save_csv(main_rows, 'exp2_主实验_node_classification.csv')
    utils.save_csv(clf_rows, 'exp2_超参_分类器.csv')
    utils.save_csv(hyp_rows, 'exp2_超参_embedding.csv')
    print('实验② 完成 ✔')


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'cora'
    run(ds)
