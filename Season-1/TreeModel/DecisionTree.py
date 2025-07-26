import numpy as np
import pandas as pd
from typing import Union, Tuple

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split: int = 2, min_samples_leaf: int = 1, task: str = 'classification'):
        self.max_depth = max_depth                                              # 最大深度
        self.min_samples_split = min_samples_split                          # 节点分裂所需的最小样本数
        self.min_samples_leaf = min_samples_leaf                            # 叶子节点所需最小的样本数
        self.tree = None
        if task not in ['classification', 'regression']:
            raise ValueError("task must be either 'classification' or 'regression'")
        self.task = task                                                    # 任务类型
        self.classes_ = None

    class Node:
        def __init__(self):
            self.feature = None
            self.threshold = None
            self.left = None
            self.right = None
            self.value = None
            self.is_leaf = False

    def _calculate_mse (self, y: np.ndarray) -> float:
        """Calculate Mean Squared Error for regression"""
        return np.mean((y - np.mean(y)) ** 2)

    """
    这段代码实现了基尼不纯度(Gini Impurity)的计算，这是决策树算法中用于衡量数据集纯度的指标
    根据基尼不纯度公式计算并返回结果：
    基尼不纯度 = 1 - Σ(p_i²)
    其中p_i是第i个类别在数据集中的概率
    如果数据集完全纯净（只有一类），基尼不纯度为0
    如果数据集完全混乱（各类别均匀分布），基尼不纯度接近1
    基尼不纯度越小，说明数据集的纯度越高，即数据集中大多数样本属于同一类别。决策树算法在分割节点时会选择能使基尼不纯度降低最多的分割方式
    """
    def _calculate_gini(self, y: np.ndarray) -> float:
        """Calculate Gini Impurity for classification"""
        _, counts = np.unique(y, return_counts=True)    #使用np.unique()函数统计y中每个唯一值的出现次数,返回一个包含唯一值的数组和该数组中每个唯一值的出现次数的元组。
        probabilities = counts / len(y)                             # 计算每个类别的概率
        return 1 - np.sum(probabilities ** 2)

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """Calculate impurity (MSE for regression, Gini for classification)"""
        if self.task == 'regression':
            return self._calculate_mse(y)
        else:
            return self._calculate_gini(y)

    '''
    实现了决策树中寻找最佳分割点的算法
    目的是找到最佳的  特征索引、阈值
    '''
    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float, float]:
        """Find the best split using appropriate criterion"""
        best_feature = None                                 # 最佳分割特征的索引
        best_threshold = None                               # 最佳分割阈值
        best_score = float('inf')                                       # 初始化最佳得分为正无穷
        n_samples, n_features = X.shape                         # 获取数据的样本数量和特征数量

        for feature in range(n_features):                       # 遍历每个特征维度
            thresholds = np.unique(X[:, feature])           # 获取当前特征列中的所有唯一值作为候选分割阈值

            for threshold in thresholds:                        # 遍历每个候选分割阈值
                left_mask = X[:, feature] <= threshold          # 创建一个布尔掩码，用于标识当前特征列中小于等于阈值的样本
                right_mask = ~left_mask                             # 创建一个布尔掩码，用于标识当前特征列中大于阈值的样本

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:                 # 检查当前分割是否满足最小样本数要求
                    continue            # 满足最小样本数要求则跳过当前分割

                left_impurity = self._calculate_impurity(y[left_mask])                  # 计算左子节点的纯度
                right_impurity = self._calculate_impurity(y[right_mask])            # 计算右子节点的纯度

                # Weighted average of impurity
                current_score = (np.sum(left_mask) * left_impurity + np.sum(right_mask) * right_impurity) / n_samples   # 计算当前分割的加权平均不纯度得分

                if current_score < best_score:          # 检查当前得分是否小于最佳得分
                    best_score = current_score          # 更新最佳得分
                    best_feature = feature                  # 更新最佳特征索引
                    best_threshold = threshold          # 更新最佳阈值

        return best_feature, best_threshold, best_score             # 返回最佳特征索引、阈值和得分

    '''
    构建决策树
    '''
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree"""
        node = self.Node()                                                      # 创建一个节点对象

        '''当决策树构建过程中满足停止条件（如达到最大深度、样本数不足、或所有样本属于同一类别）时，
        将当前节点设为叶节点，并将其预测值设置为当前样本中出现频率最高的类别'''
        # Check stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
                (len(y) < self.min_samples_split) or \
                (len(np.unique(y)) == 1):
            node.is_leaf = True
            if self.task == 'regression':
                node.value = np.mean(y)
            else:
                node.value = self.classes_[np.argmax(np.bincount(y))]                               # 返回出现次数最多的类
            return node

        # Find the best split
        feature, threshold, score = self._find_best_split(X, y)             # 寻找最佳分割点

        if feature is None:  # No valid split found
            node.is_leaf = True
            node.value = np.mean(y)
            return node

        # Split the data
        left_mask = X[:, feature] <= threshold              # 创建一个布尔掩码，用于标识当前特征列中小于等于阈值的样本
        right_mask = ~left_mask                             # 创建一个布尔掩码，用于标识当前特征列中大于阈值的样本

        # Create child nodes
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)                           # 递归调用_build_tree方法，构建左子树和右子树
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)                  # 递归调用_build_tree方法，构建左子树和右子树

        return node

    '''
    1、数据预处理：将pandas数据转换为numpy数组
    2、分类任务特殊处理：保存类别信息并将标签映射为连续整数
    3、调用递归构建树的方法，从根节点开始构建整个决策树
    4、返回训练好的模型
    '''
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """Train the decision tree"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if self.task == 'classification':
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)

        self.tree = self._build_tree(X, y, depth=0)
        return self

    '''
    1、从根节点开始，检查当前节点是否为叶节点
    2、如果是叶节点，返回该节点的预测值
    3、如果不是叶节点，根据当前样本在该节点分割特征上的值与阈值的比较结果：
        值 ≤ 阈值 → 走左子树
        值 > 阈值 → 走右子树
    4、重复上述过程，直到到达叶节点并返回预测结果
    '''
    def _predict_single(self, x: np.ndarray, node: Node) -> Union[float, str]:
        """Predict for a single sample"""
        if node.is_leaf:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)

    '''
    决策树预测入口
    '''
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict for multiple samples"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        return np.array([self._predict_single(x, self.tree) for x in X])