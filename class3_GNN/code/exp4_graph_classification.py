# -*- coding: utf-8 -*-
"""
exp4_graph_classification.py —— 实验④  GNN · 图分类 (Graph classification)

任务要点:
  - 数据集: ENZYMES(可换 DD / COLLAB), 用 dgl.data 导入
  - 将图集合划分为训练集/测试集
  - 训练 GNN(GCN / GAT) + Pooling(Mean(即 Average) / DiffPool) 模型
  - 在测试集上评测 Accuracy, 比较不同组合与训练/测试差异
  - 超参实验: GNN 层数 {1,2,3}
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import utils
from models import GraphGNN, DiffPoolModel

HIDDEN = 32


def load_graph_dataset(name='ENZYMES'):
    """加载图分类数据集, 返回 (graphs列表, labels, num_classes, in_feats)
    优先读取 TU 基准数据集官网格式(data/ENZYMES/ENZYMES/), 不支持则回退到 GINDataset.
    说明: DGL 的 GINDataset 捆绑包(dataset.zip)实际不含 ENZYMES, 故 ENZYMES 单独下载."""
    data_root = utils.DATA_DIR
    if os.path.isdir(os.path.join(data_root, name)):
        graphs, labels, num_classes, in_feats = load_tu_format(name, data_root)
    else:
        ds = dgl.data.GINDataset(name, self_loop=True)
        graphs, labels = [], []
        for g, lab in ds:
            graphs.append(g)
            labels.append(lab)
        labels = np.array(labels)
        num_classes = ds.num_classes
        in_feats = graphs[0].ndata['feat'].shape[1]
    print(f'数据集 {name}: {len(graphs)} 个图, 类别数 {num_classes}, 节点特征维 {in_feats}')
    return graphs, labels, num_classes, in_feats


def load_tu_format(name, data_root):
    """读取 TU 基准数据集官网格式: *_A.txt / *_graph_indicator.txt /
    *_graph_labels.txt / *_node_attributes.txt"""
    # 自动定位 *_A.txt 所在目录(兼容不同嵌套层级)
    dirs = [os.path.join(data_root, name),
            os.path.join(data_root, name, name),
            os.path.join(data_root, name, name, name)]
    base = None
    for d in dirs:
        cand = os.path.join(d, name + '_A.txt')
        if os.path.isfile(cand):
            base = cand[:-6]  # 去掉 "_A.txt"
            break
    if base is None:
        raise FileNotFoundError(f'未找到 {name}_A.txt, 请先下载 {name} 数据集')
    edge_list = np.loadtxt(base + '_A.txt', dtype=int, delimiter=',')
    graph_indicator = np.loadtxt(base + '_graph_indicator.txt', dtype=int) - 1
    graph_labels = (np.loadtxt(base + '_graph_labels.txt', dtype=int) - 1).astype(np.int64)
    node_attrs = np.loadtxt(base + '_node_attributes.txt', dtype=float, delimiter=',')
    num_graphs = len(graph_labels)
    src_all, dst_all = edge_list[:, 0] - 1, edge_list[:, 1] - 1
    graphs = []
    for gid in range(num_graphs):
        node_ids = np.where(graph_indicator == gid)[0]
        node_map = {int(old): new for new, old in enumerate(node_ids)}
        keep = np.isin(src_all, node_ids) & np.isin(dst_all, node_ids)
        src = np.array([node_map[int(x)] for x in src_all[keep]], dtype=np.int64)
        dst = np.array([node_map[int(x)] for x in dst_all[keep]], dtype=np.int64)
        g = dgl.graph((src, dst), num_nodes=len(node_ids))
        # TU 数据集为无向图, 补上反向边; 加自环避免 0 度节点(GCN 归一化)问题
        g = dgl.to_bidirected(g)
        g = dgl.add_self_loop(g)
        g.ndata['feat'] = torch.tensor(node_attrs[node_ids], dtype=torch.float32)
        graphs.append(g)
    num_classes = int(graph_labels.max()) + 1
    in_feats = node_attrs.shape[1]
    return graphs, graph_labels, num_classes, in_feats


def train_eval(model, train_loader, test_loader, train_g, test_g, epochs=150,
               lr=0.01, weight_decay=5e-4, device='cpu', with_aux=False):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_test = 0.0
    best_train = 0.0
    loss_hist = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batched_g, batched_y in train_loader:
            batched_g = batched_g.to(device)
            batched_y = batched_y.to(device).long()
            opt.zero_grad()
            if with_aux:
                logits, aux = model(batched_g, batched_g.ndata['feat'])
                loss = F.cross_entropy(logits, batched_y) + 0.1 * aux
            else:
                logits = model(batched_g, batched_g.ndata['feat'])
                loss = F.cross_entropy(logits, batched_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * batched_g.batch_size
        loss_hist.append(epoch_loss / max(len(train_g), 1))

        model.eval()
        with torch.no_grad():
            train_acc = predict_acc(model, train_loader, device, with_aux)
            test_acc = predict_acc(model, test_loader, device, with_aux)
        if test_acc > best_test:
            best_test = test_acc
            best_train = train_acc
    return best_train, best_test, loss_hist


def predict_acc(model, loader, device='cpu', with_aux=False):
    preds, ys = [], []
    for batched_g, batched_y in loader:
        batched_g = batched_g.to(device)
        if with_aux:
            logits, _ = model(batched_g, batched_g.ndata['feat'])
        else:
            logits = model(batched_g, batched_g.ndata['feat'])
        preds.append(logits.argmax(1).cpu().numpy())
        ys.append(batched_y.long().numpy())
    return accuracy_score(np.concatenate(ys), np.concatenate(preds))


def run(dataset='ENZYMES', seed=42):
    utils.set_seed(seed)
    print(f'===== 实验④ GNN 图分类 (数据集: {dataset}) =====')
    device = 'cpu'
    graphs, labels, num_classes, in_feats = load_graph_dataset(dataset)

    # 按图划分训练/测试集 (80/20, 分层)
    idx = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(idx, test_size=0.2,
                                           random_state=seed, stratify=labels)
    train_g, test_g = [graphs[i] for i in train_idx], [graphs[i] for i in test_idx]
    train_y, test_y = labels[train_idx], labels[test_idx]
    print(f'训练图 {len(train_g)}, 测试图 {len(test_g)}')

    train_loader = GraphDataLoader(list(zip(train_g, train_y)), batch_size=32,
                                   shuffle=True, drop_last=False, num_workers=0)
    test_loader = GraphDataLoader(list(zip(test_g, test_y)), batch_size=64,
                                  shuffle=False, num_workers=0)

    # 1) 主实验: GNN × Pooling 组合 (2层)
    main_rows = []
    configs = [
        ('GCN + MeanPool', GraphGNN('gcn', in_feats, HIDDEN, num_classes, 2, 'mean')),
        ('GAT + MeanPool', GraphGNN('gat', in_feats, HIDDEN, num_classes, 2, 'mean')),
        ('GCN + DiffPool', DiffPoolModel(in_feats, HIDDEN, num_classes, 2,
                                         num_clusters=8)),
    ]
    loss_curves = {}
    for name, model in configs:
        with_aux = 'DiffPool' in name
        train_acc, test_acc, loss_hist = train_eval(
            model, train_loader, test_loader, train_g, test_g, with_aux=with_aux)
        main_rows.append({'模型': name, '层数': 2,
                          '训练集Acc': round(train_acc, 4),
                          '测试集Acc': round(test_acc, 4)})
        loss_curves[name] = loss_hist
        print(f"  {name:16s} 训练Acc {train_acc:.4f}  测试Acc {test_acc:.4f}")
    utils.save_csv(main_rows, 'exp4_主实验_图分类.csv')
    utils.plot_training_curves(loss_curves, '图分类各模型训练损失曲线',
                               'exp4_loss_curves.png')

    # 2) 超参实验: GNN 层数 {1,2,3} (GCN+MeanPool 与 GCN+DiffPool)
    layer_rows = []
    for n_layers in [1, 2, 3]:
        for name, model in [
            (f'GCN+Mean-{n_layers}层', GraphGNN('gcn', in_feats, HIDDEN, num_classes, n_layers, 'mean')),
            (f'GCN+DiffPool-{n_layers}层', DiffPoolModel(in_feats, HIDDEN, num_classes, n_layers, num_clusters=8)),
        ]:
            with_aux = 'DiffPool' in name
            train_acc, test_acc, _ = train_eval(
                model, train_loader, test_loader, train_g, test_g, with_aux=with_aux)
            layer_rows.append({'模型': 'GCN+DiffPool' if 'DiffPool' in name else 'GCN+Mean',
                               '层数': n_layers, '训练集Acc': round(train_acc, 4),
                               '测试集Acc': round(test_acc, 4)})
            print(f"  {name:20s} 训练Acc {train_acc:.4f}  测试Acc {test_acc:.4f}")
    utils.save_csv(layer_rows, 'exp4_超参_层数.csv')
    print('实验④ 完成 ✔')


if __name__ == '__main__':
    ds = sys.argv[1] if len(sys.argv) > 1 else 'ENZYMES'
    run(ds)
