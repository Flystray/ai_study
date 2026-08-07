# -*- coding: utf-8 -*-
"""
models.py —— 本次 GNN 课程实验涉及的全部模型定义

包含:
  1. SkipGram        : 带负采样的 skip-gram 模型(DeepWalk / node2vec 共用训练目标)
  2. train_skipgram  : 给定随机游走序列, 训练 skip-gram 得到节点 embedding
  3. LINE            : 一阶/二阶近似的 LINE 模型
  4. train_line      : 训练 LINE
  5. GCN / GAT / SAGE: 半监督节点分类用的三个图卷积模型
  6. GraphGNN        : 图分类用的 GCN/GAT + 全局池化
  7. DiffPoolModel   : 图分类用的 DiffPool 模型
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl


# ============================================================================
# 1. Skip-gram 模型(DeepWalk / node2vec 共用)
# ============================================================================
class SkipGram(nn.Module):
    def __init__(self, num_nodes, emb_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim
        # 中心节点 embedding 与 上下文节点 embedding 各一套(标准 word2vec 做法)
        self.emb = nn.Embedding(num_nodes, emb_dim)
        self.ctx = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.ctx.weight, mean=0.0, std=0.1)

    def forward(self, centers, contexts, neg_contexts):
        """中心词 + 正上下文 + 负采样上下文, 返回平均 NEG 损失"""
        u = self.emb(centers)                       # [B, D]
        v = self.ctx(contexts)                      # [B, D]
        pos = torch.sum(u * v, dim=1)               # [B]
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-7)

        vneg = self.ctx(neg_contexts)               # [B, K, D]
        neg = torch.bmm(vneg, u.unsqueeze(-1)).squeeze(-1)   # [B, K]
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg) + 1e-7).sum(dim=1)

        return (pos_loss + neg_loss).mean()


# 从随机游走序列中向量化地构造 (中心, 上下文) 正样本对
def build_pairs(walk_arrays, window=5):
    """
    walk_arrays: list of np.ndarray, 每个形状为 [B, L+1](含起始节点)
    返回 (centers, contexts) 两个 np.ndarray
    """
    all_c, all_t = [], []
    for walks in walk_arrays:
        B, L = walks.shape
        cs, ts = [], []
        for off in range(1, window + 1):
            c = walks[:, :-off].reshape(-1)
            t = walks[:, off:].reshape(-1)
            cs.append(c)
            ts.append(t)
            c = walks[:, off:].reshape(-1)
            t = walks[:, :-off].reshape(-1)
            cs.append(c)
            ts.append(t)
        all_c.append(np.concatenate(cs))
        all_t.append(np.concatenate(ts))
    return np.concatenate(all_c), np.concatenate(all_t)


def train_skipgram(walks, num_nodes, emb_dim=128, window=5, negative=5,
                   epochs=3, lr=0.01, batch_size=16384, device='cpu', seed=0):
    """
    在给定随机游走上训练 skip-gram.
    walks: list of np.ndarray(每批游走的 shape [B, L+1])
    返回: (node_embedding np.ndarray [num_nodes, emb_dim], 训练loss历史)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    centers, contexts = build_pairs(walks, window)
    num_pairs = len(centers)
    model = SkipGram(num_nodes, emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # 缓存为正张量
    centers_t = torch.from_numpy(centers.astype(np.int64)).to(device)
    contexts_t = torch.from_numpy(contexts.astype(np.int64)).to(device)
    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(num_pairs, device=device)
        epoch_loss = 0.0
        n_batch = 0
        for i in range(0, num_pairs, batch_size):
            idx = perm[i:i + batch_size]
            c = centers_t[idx]
            t = contexts_t[idx]
            neg = torch.randint(0, num_nodes, (len(idx), negative), device=device)
            opt.zero_grad()
            loss = model(c, t, neg)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batch += 1
        losses.append(epoch_loss / max(n_batch, 1))
    emb = model.emb.weight.detach().cpu().numpy()
    return emb, losses


# ============================================================================
# 2. LINE 模型(一阶 / 二阶近似 + 负采样)
# ============================================================================
class LINE(nn.Module):
    def __init__(self, num_nodes, emb_dim, order='second'):
        super().__init__()
        self.order = order
        self.emb = nn.Embedding(num_nodes, emb_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.1)
        if order == 'second':
            self.ctx = nn.Embedding(num_nodes, emb_dim)
            nn.init.normal_(self.ctx.weight, mean=0.0, std=0.1)

    def forward(self, src, dst, neg_dst):
        # 一阶: 使用同一套 embedding; 二阶: 源用 emb, 目标(上下文)用 ctx
        if self.order == 'first':
            u = self.emb(src)                        # [B, D]
            v = self.emb(dst)                        # [B, D]
        else:
            u = self.emb(src)
            v = self.ctx(dst)
        pos = torch.sum(u * v, dim=1)                # [B]
        pos_loss = -torch.log(torch.sigmoid(pos) + 1e-7)

        if self.order == 'first':
            vneg = self.emb(neg_dst)                 # [B, K, D]
        else:
            vneg = self.ctx(neg_dst)
        neg = torch.bmm(vneg, u.unsqueeze(-1)).squeeze(-1)   # [B, K]
        neg_loss = -torch.log(1.0 - torch.sigmoid(neg) + 1e-7).sum(dim=1)
        return (pos_loss + neg_loss).mean()


def train_line(g, num_nodes, emb_dim=128, order='second', negative=5,
               epochs=10, lr=0.01, batch_size=512, device='cpu', seed=0,
               edge_budget=None):
    """
    在图上训练 LINE. 直接采样图上的边作为正样本, 随机采样负节点.
    g: DGL 图(训练图, 已去掉测试边)
    注意: 边数通常较少, 因此默认 batch_size 保持较小, 保证每个 epoch 有多步梯度更新;
         二阶损失参数更多(两套 embedding), 需要较多 epoch 才能收敛, 默认 10.
    返回: node_embedding [num_nodes, emb_dim], loss历史
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    edges = g.edges()
    src = edges[0].numpy()
    dst = edges[1].numpy()
    # 去掉自环 (对链接预测无意义)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if edge_budget is not None and len(src) > edge_budget:
        keep = np.random.choice(len(src), edge_budget, replace=False)
        src, dst = src[keep], dst[keep]

    model = LINE(num_nodes, emb_dim, order=order).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    src_t = torch.from_numpy(src.astype(np.int64)).to(device)
    dst_t = torch.from_numpy(dst.astype(np.int64)).to(device)

    losses = []
    num_edges = len(src_t)
    for epoch in range(epochs):
        perm = torch.randperm(num_edges, device=device)
        ep_loss = 0.0
        n_batch = 0
        for i in range(0, num_edges, batch_size):
            idx = perm[i:i + batch_size]
            s = src_t[idx]
            d = dst_t[idx]
            neg = torch.randint(0, num_nodes, (len(idx), negative), device=device)
            opt.zero_grad()
            loss = model(s, d, neg)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_batch += 1
        losses.append(ep_loss / max(n_batch, 1))
    emb = model.emb.weight.detach().cpu().numpy()
    return emb, losses


# ============================================================================
# 3. 半监督节点分类模型: GCN / GAT / GraphSAGE
# ============================================================================
class GCN(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(dgl.nn.GraphConv(in_feats, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.GraphConv(hidden, hidden))
        self.convs.append(dgl.nn.GraphConv(hidden, out_feats))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, g, x):
        for i, conv in enumerate(self.convs):
            x = conv(g, x)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x


class GAT(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_layers=2, num_heads=4,
                 dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dgl.nn.GATConv(in_feats, out_feats, num_heads=1))
            self.last_head = 1
            self.last_concat = False
        else:
            self.layers.append(dgl.nn.GATConv(in_feats, hidden, num_heads=num_heads))
            for _ in range(num_layers - 2):
                self.layers.append(dgl.nn.GATConv(hidden * num_heads, hidden,
                                                  num_heads=num_heads))
            self.layers.append(dgl.nn.GATConv(hidden * num_heads, out_feats,
                                              num_heads=1))
            self.last_head = 1
            self.last_concat = False
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            x = layer(g, x)
            if i < len(self.layers) - 1:
                x = x.flatten(1)
                x = self.act(x)
                x = self.dropout(x)
            else:
                x = x.mean(1)   # 最后一层多头取平均
        return x


class SAGE(nn.Module):
    def __init__(self, in_feats, hidden, out_feats, num_layers=2, dropout=0.5,
                 aggregator='mean'):
        super().__init__()
        agg = {'mean': 'mean', 'gcn': 'gcn', 'pool': 'pool'}[aggregator]
        self.convs = nn.ModuleList()
        self.convs.append(dgl.nn.SAGEConv(in_feats, hidden, agg))
        for _ in range(num_layers - 2):
            self.convs.append(dgl.nn.SAGEConv(hidden, hidden, agg))
        self.convs.append(dgl.nn.SAGEConv(hidden, out_feats, agg))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, g, x):
        for i, conv in enumerate(self.convs):
            x = conv(g, x)
            if i < len(self.convs) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x


# ============================================================================
# 4. 图分类模型: GCN/GAT + 全局池化
# ============================================================================
class GraphGNN(nn.Module):
    """GCN/GAT + 全局池化(mean/max/sum) + 线性分类头, 用于图分类"""
    def __init__(self, gnn_type, in_feats, hidden, out_feats, num_layers=2,
                 pool='mean', dropout=0.5, num_heads=4):
        super().__init__()
        self.gnn_type = gnn_type
        self.pool = pool
        if gnn_type == 'gcn':
            self.backbone = GCN(in_feats, hidden, hidden, num_layers, dropout)
            hidden_final = hidden
        elif gnn_type == 'gat':
            self.backbone = GAT(in_feats, hidden, hidden, num_layers,
                                num_heads=num_heads, dropout=dropout)
            hidden_final = hidden
        else:
            raise ValueError(gnn_type)
        self.classifier = nn.Linear(hidden_final, out_feats)

    def forward(self, g, x):
        h = self.backbone(g, x)
        # DGL 2.2 的全局池化需要 ndata 特征名
        g.ndata['_readout'] = h
        if self.pool == 'mean':
            h = dgl.mean_nodes(g, '_readout')
        elif self.pool == 'max':
            h = dgl.max_nodes(g, '_readout')
        elif self.pool == 'sum':
            h = dgl.sum_nodes(g, '_readout')
        else:
            raise ValueError(self.pool)
        return self.classifier(h)


# ============================================================================
# 5. 图分类模型: DiffPool(手写实现)
# ============================================================================
class DiffPoolLayer(nn.Module):
    """
    DiffPool 层(DGL 2.2.1 包未内置, 这里依据论文原意实现).
    学习分配矩阵 S, 把 N 个节点粗化为 K 个簇:
      S = softmax(GNN_assign(X, A))        [N, K]
      Z = GNN_embed(X, A)                  [N, d]
      X' = S^T Z                           [K, d]   粗化后的节点特征
      A' = S^T A S                         [K, K]   粗化后的邻接矩阵
      辅助损失 = ||A'||_F(稠密链接预测) + 熵正则 H(S)
    """
    def __init__(self, in_feats, hidden_feats, num_clusters):
        super().__init__()
        self.num_clusters = num_clusters
        self.assign_gnn = dgl.nn.GraphConv(in_feats, num_clusters)
        self.embed_gnn = dgl.nn.GraphConv(in_feats, hidden_feats)

    def forward(self, g, h):
        S = torch.softmax(self.assign_gnn(g, h), dim=1)        # [N, K]
        Z = self.embed_gnn(g, h)                               # [N, d]
        # DGL 2.2 的 adjacency_matrix() 返回 dgl SparseMatrix, 转稠密(批处理图为块对角)
        A = g.adjacency_matrix().to_dense().to(torch.float32)  # [N, N]

        # 逐图进行粗化, 避免就地赋值破坏梯度回传
        bn = g.batch_num_nodes()
        G = len(bn)
        Z_blocks, link_loss_terms, ent_sum = [], [], 0.0
        start = 0
        for n in bn:
            n = int(n)
            Sg = S[start:start + n]                            # [n, K]
            Zg = Z[start:start + n]                            # [n, d]
            Ag = A[start:start + n, start:start + n]           # [n, n]
            # 粗化特征: 加权和 + 按簇内节点权重归一化(等价带权平均, 避免数值放大)
            Xg = (Sg.t() @ Zg) / (Sg.sum(dim=0, keepdim=True).t() + 1e-8)   # [K, d]
            Ag_new = Sg.t() @ Ag @ Sg                          # [K, K] 粗化邻接
            Z_blocks.append(Xg)
            link_loss_terms.append((Ag_new ** 2).mean())
            ent_sum = ent_sum + (-(Sg * torch.log(Sg + 1e-7)).sum(dim=1).mean())
            start += n

        Z_new = torch.cat(Z_blocks, dim=0)                     # [G*K, d]
        aux_loss = (sum(link_loss_terms) / G) + ent_sum / G
        return Z_new, aux_loss


class DiffPoolModel(nn.Module):
    """
    图分类模型: 预编码GCN -> DiffPool 粗化 -> 全局平均池化 -> 分类头
    """
    def __init__(self, in_feats, hidden, out_feats, num_layers=2, dropout=0.5,
                 num_clusters=8):
        super().__init__()
        self.embed_net = GCN(in_feats, hidden, hidden, num_layers, dropout)
        # 归一化节点特征, 防止 DiffPool 的 S^T Z 加权和数值爆炸
        self.node_norm = nn.LayerNorm(hidden)
        self.diffpool = DiffPoolLayer(hidden, hidden, num_clusters)
        # 全局跳跃连接: 拼接原始图均值池化 与 粗化后均值池化, 增强稳健性
        self.classifier = nn.Linear(2 * hidden, out_feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = self.embed_net(g, x)
        h = self.node_norm(h)
        # 原始图上的全局均值池化(跳跃连接)
        g.ndata['_orig'] = h
        gp_orig = dgl.mean_nodes(g, '_orig')                   # [B, d]
        pooled_feat, aux_loss = self.diffpool(g, h)            # [G*K, d]
        # 每个粗化图固定 K 个节点, 直接 reshape 做均值池化
        K = self.diffpool.num_clusters
        gp_pooled = pooled_feat.view(g.batch_size, K, -1).mean(dim=1)   # [B, d]
        gp = torch.cat([gp_orig, gp_pooled], dim=1)            # [B, 2d]
        gp = self.dropout(F.relu(gp))
        logits = self.classifier(gp)
        return logits, aux_loss
