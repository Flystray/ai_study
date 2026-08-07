"""
两层神经网络 (一个隐藏层) - 手写数字识别
====================================
结构: 输入层(64) → 隐藏层(n_hidden, Sigmoid) → 输出层(10, Softmax)
损失: 交叉熵损失 (Cross-Entropy Loss)
优化: 批量梯度下降 (Batch Gradient Descent)
"""

import numpy as np


class TwoLayerNN:
    """
    两层神经网络分类器

    参数
    ----------
    n_hidden : int
        隐藏层神经元数量
    learning_rate : float
        梯度下降学习率
    n_iter : int
        迭代次数
    verbose : bool
        是否打印训练过程
    """

    def __init__(self, n_hidden=64, learning_rate=0.1, n_iter=1000,
                 batch_size=None, verbose=True):
        """
        参数
        ----------
        batch_size : int or None
            批量大小。None 或 ≥ 总样本数时 = 全批量梯度下降。
            较小的 batch_size (如 32/64) = Mini-batch 梯度下降。
        """
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.loss_history = []

        # 网络参数 (在 fit 时初始化)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _initialize_parameters(self, n_input, n_output):
        """
        Xavier 初始化参数
        """
        # 隐藏层: W1 ∈ R^(n_hidden × n_input), b1 ∈ R^(n_hidden × 1)
        limit1 = np.sqrt(6 / (n_input + self.n_hidden))
        self.W1 = np.random.uniform(-limit1, limit1, (self.n_hidden, n_input))
        self.b1 = np.zeros((self.n_hidden, 1))

        # 输出层: W2 ∈ R^(n_output × n_hidden), b2 ∈ R^(n_output × 1)
        limit2 = np.sqrt(6 / (self.n_hidden + n_output))
        self.W2 = np.random.uniform(-limit2, limit2, (n_output, self.n_hidden))
        self.b2 = np.zeros((n_output, 1))

    @staticmethod
    def _sigmoid(Z):
        """Sigmoid 激活函数"""
        # 用 np.clip 防止 exp 溢出
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def _sigmoid_derivative(A):
        """Sigmoid 的导数: σ(x) * (1 - σ(x))"""
        return A * (1 - A)

    @staticmethod
    def _softmax(Z):
        """
        Softmax 激活函数（数值稳定版）
        输入 Z: (n_classes, m)
        输出 A: (n_classes, m)，每列和为 1
        """
        # 减去最大值防止指数溢出
        Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward(self, X):
        """
        前向传播

        参数
        ----------
        X : ndarray, shape (n_input, m)
            输入数据，每列一个样本

        返回
        -------
        cache : dict
            包含 Z1, A1, Z2, A2
        """
        # 输入层 → 隐藏层
        Z1 = self.W1 @ X + self.b1          # (n_hidden, m)
        A1 = self._sigmoid(Z1)               # (n_hidden, m)

        # 隐藏层 → 输出层
        Z2 = self.W2 @ A1 + self.b2          # (n_output, m)
        A2 = self._softmax(Z2)               # (n_output, m)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return cache

    def backward(self, X, Y, cache):
        """
        反向传播

        参数
        ----------
        X : ndarray, shape (n_input, m)
        Y : ndarray, shape (n_output, m)
            one-hot 编码的标签
        cache : dict
            前向传播的中间结果

        返回
        -------
        grads : dict
            包含 dW1, db1, dW2, db2
        """
        A1 = cache["A1"]
        A2 = cache["A2"]
        m = X.shape[1]  # 样本数

        # --- 输出层梯度 ---
        # dL/dZ2 = A2 - Y  (Softmax + Cross-Entropy 的简化结果)
        dZ2 = A2 - Y                                     # (n_output, m)
        dW2 = (1 / m) * (dZ2 @ A1.T)                     # (n_output, n_hidden)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (n_output, 1)

        # --- 隐藏层梯度 ---
        dA1 = self.W2.T @ dZ2                             # (n_hidden, m)
        dZ1 = dA1 * self._sigmoid_derivative(A1)             # (n_hidden, m)
        dW1 = (1 / m) * (dZ1 @ X.T)                       # (n_hidden, n_input)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)    # (n_hidden, 1)

        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def compute_loss(self, A2, Y):
        """
        计算交叉熵损失

        L = -(1/m) * Σ Σ Y * log(A2 + ε)

        参数
        ----------
        A2 : ndarray, shape (n_output, m)
            Softmax 输出
        Y : ndarray, shape (n_output, m)
            one-hot 标签

        返回
        -------
        loss : float
        """
        m = Y.shape[1]
        # 加极小值防止 log(0)
        eps = 1e-8
        loss = -(1 / m) * np.sum(Y * np.log(A2 + eps))
        return loss

    def predict(self, X):
        """
        
        预测类别

        参数
        ----------
        X : ndarray, shape (n_input, m) 或 (n_input,)

        返回
        -------
        y_pred : ndarray, shape (m,)  每个样本的预测类别 (0-9)
        """
        # 如果 X 是 1-D，reshape 成 (n_input, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        # 如果 X 是 (m, n_input)，转置为 (n_input, m)
        if X.ndim == 2 and X.shape[0] != self.W1.shape[1]:
            X = X.T

        cache = self.forward(X)
        A2 = cache["A2"]
        y_pred = np.argmax(A2, axis=0)
        return y_pred

    def score(self, X, y):
        """
        计算分类准确率

        参数
        ----------
        X : ndarray, shape (m, n_input) 或 (n_input, m)
        y : ndarray, shape (m,)  类别标签 0-9

        返回
        -------
        acc : float  准确率 (0~1)
        """
        # 处理 X 形状: 内部使用 (n_input, m)
        if X.ndim == 2 and X.shape[0] != self.W1.shape[1]:
            X = X.T  # (m, n_input) → (n_input, m)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return acc

    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练神经网络

        参数
        ----------
        X : ndarray, shape (m, n_input)  训练数据
        y : ndarray, shape (m,)          训练标签
        X_val : ndarray, optional         验证数据
        y_val : ndarray, optional         验证标签
        """
        m, n_input = X.shape
        n_output = len(np.unique(y))

        # 转换为内部列向量格式: (n, m)
        X, y_orig = X.T, y
        if X_val is not None:
            X_val = X_val.T

        # 初始化参数
        self._initialize_parameters(n_input, n_output)

        # one-hot 编码标签
        Y = np.eye(n_output)[y_orig].T  # (n_output, m)

        # 判断是否为 mini-batch 模式
        use_minibatch = self.batch_size is not None and self.batch_size < m
        if use_minibatch:
            n_batches = max(1, m // self.batch_size)
            if self.verbose:
                print(f"Mini-batch 模式: batch_size={self.batch_size}, "
                      f"每轮 {n_batches} 个 batch")

        self.loss_history = []
        val_acc_history = []

        for i in range(self.n_iter):
            if use_minibatch:
                # --- Mini-batch: 每轮跑所有 batch ---
                # 打乱数据
                shuffle_idx = np.random.permutation(m)
                X_shuffled = X[:, shuffle_idx]
                Y_shuffled = Y[:, shuffle_idx]

                epoch_loss = 0
                for b in range(n_batches):
                    start = b * self.batch_size
                    end = min(start + self.batch_size, m)
                    X_batch = X_shuffled[:, start:end]
                    Y_batch = Y_shuffled[:, start:end]

                    # 前向 → 反向 → 更新
                    cache = self.forward(X_batch)
                    grads = self.backward(X_batch, Y_batch, cache)
                    self._update_params(grads)

                    # 累计损失（按 batch 大小加权）
                    batch_loss = self.compute_loss(cache["A2"], Y_batch)
                    epoch_loss += batch_loss * (end - start) / m

                self.loss_history.append(epoch_loss)
                loss = epoch_loss
            else:
                # --- 全批量梯度下降（原始逻辑）---
                cache = self.forward(X)
                loss = self.compute_loss(cache["A2"], Y)
                self.loss_history.append(loss)
                grads = self.backward(X, Y, cache)
                self._update_params(grads)

            # --- 记录验证集准确率 ---
            if X_val is not None and y_val is not None:
                val_acc = self.score(X_val, y_val)
                val_acc_history.append(val_acc)

            # --- 打印进度 ---
            if self.verbose and (i + 1) % 100 == 0:
                msg = f"Iter {i+1:4d}/{self.n_iter} | Loss: {loss:.4f}"
                if val_acc_history:
                    msg += f" | Val Acc: {val_acc_history[-1]:.4f}"
                print(msg)

        return val_acc_history if val_acc_history else None

    def _update_params(self, grads):
        """梯度下降更新参数"""
        self.W2 -= self.learning_rate * grads["dW2"]
        self.b2 -= self.learning_rate * grads["db2"]
        self.W1 -= self.learning_rate * grads["dW1"]
        self.b1 -= self.learning_rate * grads["db1"]
