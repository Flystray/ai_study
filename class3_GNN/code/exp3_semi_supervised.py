# -*- coding: utf-8 -*-
"""
exp3_semi_supervised.py —— 实验③  GNN · 半监督节点分类

任务要点:
  - 数据集: Cora(可换 citeseer / pubmed), 加载节点属性与节点标签
  - 给定图 G=(V, X, A, Y), 仅用带标签节点训练 GNN, 预测无标签节点标签
  - 卷积层使用 dgl.nn 下的 GCNConv / GATConv / SAGEConv
  - 评测指标 Accuracy (训练集 / 测试集)
  - 超参实验: GNN 层数 {1,2,3}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import accuracy_score

import utils
from models import GCN, GAT, SAGE


def train_model(model, g, features, train_mask, val_mask, epochs=200,
                lr=0.01, weight_decay=5e-4, patience=20, device='cpu'):
    """带 early stopping 的标准半监督训练, 返回 (最佳测试acc, 训练acc, 损失历史)"""
    model = model.to(device)
    g = g.to(device)
    features = features.to(device)
    labels = g.ndata['label'].to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = 0.0
    best_test = 0.0
    best_train = 0.0
    wait = 0
    loss_hist = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())
        model.eval()
        with torch.no_grad():
            logits_eval = model(g, features)
            pred = logits_eval.argmax(1)
            train_acc = accuracy_score(labels[train_mask].cpu().numpy(),
                                       pred[train_mask].cpu().numpy())
            val_acc = accuracy_score(labels[val_mask].cpu().numpy(),
                                     pred[val_mask].cpu().numpy())
            test_acc = accuracy_score(labels[g.ndata['test_mask']].cpu().numpy(),
                                      pred[g.ndata['test_mask']].cpu().numpy())
        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc
            best_train = train_acc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_train, best_test, loss_hist


def run(dataset='cora', seed=42):
    utils.set_seed(seed)
    print(f'===== 实验③ GNN 半监督节点分类 (数据集: {dataset}) =====')

    g, features, labels, num_classes, num_nodes = utils.load_node_dataset(dataset)
    # GCN 等模型需要自环(含节点自身信息), 统一添加自环
    g = dgl.add_self_loop(g)
    in_feats = features.shape[1]
    hidden = 32
    device = 'cpu'
    print(f'节点数 {num_nodes}, 特征维 {in_feats}, 类别数 {num_classes}')
    print(f"训练节点 {int(g.ndata['train_mask'].sum())}, "
          f"验证节点 {int(g.ndata['val_mask'].sum())}, "
          f"测试节点 {int(g.ndata['test_mask'].sum())}")

    # 1) 主实验: GCN / GAT / SAGE, 2 层
    main_rows = []
    loss_curves = {}
    for name, builder in [('GCN', lambda n: GCN(in_feats, hidden, num_classes, n)),
                          ('GAT', lambda n: GAT(in_feats, hidden, num_classes, n)),
                          ('SAGE', lambda n: SAGE(in_feats, hidden, num_classes, n))]:
        model = builder(2)
        train_acc, test_acc, loss_hist = train_model(
            model, g, features, g.ndata['train_mask'], g.ndata['val_mask'])
        loss_curves[name] = loss_hist
        main_rows.append({'模型': name, '层数': 2,
                          '训练集Acc': round(train_acc, 4),
                          '测试集Acc': round(test_acc, 4)})
        print(f"  {name:6s} 训练Acc {train_acc:.4f}  测试Acc {test_acc:.4f}")

    utils.save_csv(main_rows, 'exp3_主实验_半监督节点分类.csv')
    utils.plot_training_curves(loss_curves, '三种 GNN 训练损失曲线 (2层)',
                               'exp3_loss_curves.png')

    # 2) 超参实验: GNN 层数 {1,2,3}
    layer_rows = []
    for n_layers in [1, 2, 3]:
        for name, builder in [('GCN', lambda n, L=n_layers: GCN(in_feats, hidden, num_classes, L)),
                              ('GAT', lambda n, L=n_layers: GAT(in_feats, hidden, num_classes, L)),
                              ('SAGE', lambda n, L=n_layers: SAGE(in_feats, hidden, num_classes, L))]:
            model = builder(n_layers)
            train_acc, test_acc, _ = train_model(
                model, g, features, g.ndata['train_mask'], g.ndata['val_mask'])
            layer_rows.append({'模型': name, '层数': n_layers,
                               '训练集Acc': round(train_acc, 4),
                               '测试集Acc': round(test_acc, 4)})
            print(f"  {name:6s} × 层数{n_layers}  训练Acc {train_acc:.4f}  测试Acc {test_acc:.4f}")

    utils.save_csv(layer_rows, 'exp3_超参_层数.csv')
    print('实验③ 完成 ✔')


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'cora'
    run(ds)
