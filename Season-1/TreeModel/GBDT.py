import numpy as np
from DecisionTree import DecisionTree

class GBDT:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2):
        """
        Initialize the Gradient Boosting Decision Tree for regression.

        Parameters:
        -----------
        n_estimators : int
            Number of boosting stages (trees) to perform
        learning_rate : float
            Learning rate shrinks the contribution of each tree
        max_depth : int
            Maximum depth of each decision tree
        min_samples_split : int
            Minimum number of samples required to split an internal node
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    '''
    GBDT工作原理总结：
        1、初始化：设定初始预测值（通常为0）
        2、迭代训练：
            计算当前模型的预测残差
            训练一棵新树来拟合这些残差
            用学习率控制的新树预测结果更新整体预测
            保存这棵树
        3、最终模型：所有树的加权组合
    这种方法通过逐步修正误差来构建强大的集成模型，每棵树都专注于解决前序模型的不足之处。
    '''
    def fit(self, X, y):
        """
        Fit the gradient boosting model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        """
        # Initialize predictions with zeros
        '''
        初始化预测值F为全零数组，与y形状相同
        在GBDT中，初始预测值通常设为0或目标值的均值
        '''
        F = np.zeros_like(y, dtype=np.float64)

        for epoch in range(self.n_estimators):      # 主循环，迭代self.n_estimators次（即树的数量）, 每次迭代添加一棵新树
            # Compute residuals
            residuals = y - F           # 计算残差（负梯度）：真实值与当前预测值的差, 这是梯度提升的核心思想：后续的树拟合前一棵树的预测误差

            # Fit a tree on residuals
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task='regression'
            )
            '''
            用当前的特征数据X和残差residuals训练这棵决策树
            这棵树的目标是学习如何修正当前的预测误差
            '''
            tree.fit(X, residuals)

            '''
            更新预测值：
                用学习率缩放新树的预测结果
                将其加到当前预测值上
                学习率控制每次更新的步长，防止过拟合
            '''
            # Update predictions
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

            # 打印训练进度和残差值
            print(f"Training iteration {epoch+1}/{self.n_estimators} - First 5 residuals: {residuals.head().values}, Residual mean: {residuals.mean():.4f}, std: {residuals.std():.4f}")

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        predictions = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

    def score(self, X, y):
        """
        Return the coefficient of determination R^2.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values

        Returns:
        --------
        score : float
            R^2 score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

