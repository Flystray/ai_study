# GNN 课程实验（class3_GNN）

基于 **DGL** 框架的图神经网络（GNN）课程实验。任务说明见 `class3_GNN.pdf`。

## 目录结构

```
class3_GNN/
├── class3_GNN.pdf        # 课程任务说明
├── 文献/                  # 相关论文（DeepWalk/LINE/node2vec/GCN/GAT/GraphSAGE/DiffPool 等）
├── code/                 # 全部实验代码
│   ├── models.py         # 模型定义（SkipGram/LINE/GCN/GAT/SAGE/GraphGNN/DiffPool）
│   ├── utils.py          # 数据加载、边划分、评估、t-SNE 可视化、结果保存
│   ├── dataset_viz.py    # 下载并可视化 Cora/Citeseer/Pubmed 三个数据集
│   ├── exp1_link_prediction.py    # 实验一：Node embedding · 链路预测
│   ├── exp2_node_classification.py# 实验二：Node embedding · 节点分类
│   ├── exp3_semi_supervised.py    # 实验三：GNN 半监督节点分类
│   ├── exp4_graph_classification.py# 实验四：GNN 图分类
│   ├── make_summary.py   # 汇总各实验 CSV 并生成对比图表
│   └── requirements.txt   # 依赖清单
├── data/                  # 数据集缓存（DGL 自动下载 + ENZYMES）
├── result/                # 实验结果：CSV 表格 + PNG 图表
└── 报告/                   # 实验报告（实验报告.md / 实验报告.html / images/）
```

## 环境安装

DGL 2.2.1 的 Windows 轮子只支持 **Python 3.12**，且其预编译 C++ 库与 **PyTorch 2.3.0** 对齐，因此使用独立 conda 环境：

```bash
conda create -n gnn python=3.12 -y
conda activate gnn
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
pip install dgl==2.2.1 numpy==1.26.4 scipy scikit-learn matplotlib networkx pandas torchdata==0.7.1 pyyaml pydantic
```

## 运行实验

```bash
# 在 code/ 同级目录下运行（数据集首次运行会自动下载到 data/）
python code/dataset_viz.py                       # 数据集概览与可视化（Cora/Citeseer/Pubmed）
python code/exp1_link_prediction.py cora        # 实验一
python code/exp2_node_classification.py cora    # 实验二
python code/exp3_semi_supervised.py cora        # 实验三
python code/exp4_graph_classification.py ENZYMES  # 实验四
python code/make_summary.py                     # 汇总各实验 CSV 并生成对比图表
```

实验结果（CSV）与图表（PNG）输出到 `result/`；完整实验报告见 `报告/实验报告.md`（另附 `实验报告.html` 可离线查看）。

## 数据集说明

- 实验一/二/三：`dgl.data.CoraGraphDataset()`（可选 Citeseer、Pubmed，运行时传入对应名称）
- 实验四：`dgl.data.GINDataset('ENZYMES')`（可选 DD、COLLAB）
