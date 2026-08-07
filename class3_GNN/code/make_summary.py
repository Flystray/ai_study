# -*- coding: utf-8 -*-
"""
make_summary.py —— 汇总各实验 CSV 结果, 生成对比图表(供报告使用)
运行方式: python make_summary.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def read_csv(name):
    path = os.path.join(utils.RESULTS_DIR, name)
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def bar_chart(labels, data_dict, title, path, ylabel='指标值', rot=0):
    x = np.arange(len(labels))
    width = 0.8 / len(data_dict)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (name, vals) in enumerate(data_dict.items()):
        bars = ax.bar(x + i * width, vals, width, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x + width * (len(data_dict) - 1) / 2)
    ax.set_xticklabels(labels, rotation=rot)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    full = os.path.join(utils.RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')


def line_chart(xs, data_dict, title, path, ylabel='指标值', xlabel=''):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    markers = ['o', 's', '^', 'D']
    for i, (name, vals) in enumerate(data_dict.items()):
        ax.plot(xs, vals, marker=markers[i % 4], label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    full = os.path.join(utils.RESULTS_DIR, path)
    fig.savefig(full, dpi=150)
    plt.close(fig)
    print(f'[图已保存] {full}')


def summarize_exp1():
    rows = read_csv('exp1_主实验_link_prediction.csv')
    models = [r['模型'] for r in rows]
    train_auc = [float(r['训练集AUC']) for r in rows]
    test_auc = [float(r['测试集AUC']) for r in rows]
    bar_chart(models, {'训练集AUC': train_auc, '测试集AUC': test_auc},
              '实验① 三种 Node Embedding 模型链路预测 AUC',
              'exp1_主实验对比.png', ylabel='AUC')

    hyp = read_csv('exp1_超参实验_link_prediction.csv')
    # DeepWalk 路径长度
    dw_wl = [r for r in hyp if r['模型'] == 'DeepWalk' and r['超参'] == '路径长度']
    xs = [int(r['值']) for r in dw_wl]
    ys = [float(r['测试AUC']) for r in dw_wl]
    line_chart(xs, {'DeepWalk': ys}, '实验① DeepWalk 路径长度对链路预测 AUC 的影响',
               'exp1_超参_路径长度.png', xlabel='路径长度')
    # DeepWalk 每节点路径数
    dw_nw = [r for r in hyp if r['模型'] == 'DeepWalk' and r['超参'] == '每节点路径数']
    xs = [int(r['值']) for r in dw_nw]
    ys = [float(r['测试AUC']) for r in dw_nw]
    line_chart(xs, {'DeepWalk': ys}, '实验① DeepWalk 每节点起始路径数的影响',
               'exp1_超参_路径数量.png', xlabel='每节点路径数')
    # node2vec p,q
    n2v = [r for r in hyp if r['模型'] == 'node2vec']
    labels = [r['超参'] for r in n2v]
    ys = [float(r['测试AUC']) for r in n2v]
    bar_chart(labels, {'node2vec': ys}, '实验① node2vec (p,q) 对链路预测 AUC 的影响',
              'exp1_超参_pq.png', ylabel='AUC', rot=30)
    # LINE 阶数 + 负样本
    line_ord = [r for r in hyp if r['模型'] == 'LINE' and r['超参'] == '损失函数(阶数)']
    labels = [r['值'] for r in line_ord]
    ys = [float(r['测试AUC']) for r in line_ord]
    bar_chart(labels, {'LINE': ys}, '实验① LINE 一阶/二阶损失对比',
              'exp1_超参_line_order.png', ylabel='AUC')
    line_neg = [r for r in hyp if r['模型'] == 'LINE' and r['超参'] == '负样本数']
    xs = [int(r['值']) for r in line_neg]
    ys = [float(r['测试AUC']) for r in line_neg]
    line_chart(xs, {'LINE': ys}, '实验① LINE 负样本数量对链路预测 AUC 的影响',
               'exp1_超参_line_neg.png', xlabel='负样本数量')


def summarize_exp2():
    rows = read_csv('exp2_主实验_node_classification.csv')
    models = [r['Embedding模型'] for r in rows]
    train_acc = [float(r['训练集Acc']) for r in rows]
    test_acc = [float(r['测试集Acc']) for r in rows]
    bar_chart(models, {'训练集Acc': train_acc, '测试集Acc': test_acc},
              '实验② 三种 Embedding 模型节点分类准确率',
              'exp2_主实验对比.png', ylabel='Accuracy')
    clf = read_csv('exp2_超参_分类器.csv')
    for m in ['DeepWalk', 'node2vec', 'LINE']:
        sub = [r for r in clf if r['Embedding模型'] == m]
        labels = [r['分类器'] for r in sub]
        ys = [float(r['测试集Acc']) for r in sub]
        bar_chart(labels, {m: ys}, f'实验② {m} 使用不同分类器的测试准确率',
                  f'exp2_分类器_{m}.png', ylabel='Accuracy')


def summarize_exp3():
    rows = read_csv('exp3_主实验_半监督节点分类.csv')
    models = [r['模型'] for r in rows]
    train_acc = [float(r['训练集Acc']) for r in rows]
    test_acc = [float(r['测试集Acc']) for r in rows]
    bar_chart(models, {'训练集Acc': train_acc, '测试集Acc': test_acc},
              '实验③ GCN/GAT/GraphSAGE 半监督节点分类准确率',
              'exp3_主实验对比.png', ylabel='Accuracy')
    lay = read_csv('exp3_超参_层数.csv')
    xs = sorted(set(int(r['层数']) for r in lay))
    data = {}
    for m in ['GCN', 'GAT', 'SAGE']:
        data[m] = [float(r['测试集Acc']) for r in lay
                   if r['模型'] == m and int(r['层数']) in xs]
    line_chart(xs, data, '实验③ GNN 层数对半监督节点分类测试准确率的影响',
               'exp3_超参_层数.png', xlabel='GNN 层数')


def summarize_exp4():
    rows = read_csv('exp4_主实验_图分类.csv')
    models = [r['模型'] for r in rows]
    train_acc = [float(r['训练集Acc']) for r in rows]
    test_acc = [float(r['测试集Acc']) for r in rows]
    bar_chart(models, {'训练集Acc': train_acc, '测试集Acc': test_acc},
              '实验④ GNN×Pooling 组合图分类准确率',
              'exp4_主实验对比.png', ylabel='Accuracy', rot=15)
    lay = read_csv('exp4_超参_层数.csv')
    for m in ['GCN+Mean', 'GCN+DiffPool']:
        sub = sorted([r for r in lay if r['模型'] == m], key=lambda r: int(r['层数']))
        xs = [int(r['层数']) for r in sub]
        ys = [float(r['测试集Acc']) for r in sub]
        line_chart(xs, {m: ys}, f'实验④ {m} 层数对图分类测试准确率的影响',
                   f'exp4_超参_{m}.png', xlabel='GNN 层数')


if __name__ == '__main__':
    summarize_exp1()
    summarize_exp2()
    summarize_exp3()
    summarize_exp4()
    print('汇总图表生成完成 ✔')
