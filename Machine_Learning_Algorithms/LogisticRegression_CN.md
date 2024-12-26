### 逻辑回归(Logistic Regression)

逻辑回归是一种用于分类问题的统计方法，尽管名字中有“回归”一词，但它主要用于二分类问题。以下是逻辑回归的推导过程：

**1. 逻辑回归的基本概念**

逻辑回归的目标是预测一个二元因变量（0或1），给定一个或多个自变量。它通过估计事件发生的概率来实现这一点。

**2. 线性模型**

首先，假设我们有一个线性模型：

$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

其中 $z$ 是线性组合，$\beta_0, \beta_1, \ldots, \beta_n$ 是模型参数，$x_1, x_2, \ldots, x_n$ 是自变量。

**3. Sigmoid 函数**

为了将线性模型的输出转换为概率，我们使用 Sigmoid 函数（也称为逻辑函数）：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Sigmoid 函数将任何实数映射到 (0, 1) 区间，因此适合表示概率。

**4. 逻辑回归模型**

逻辑回归模型可以表示为：

$$
P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
$$

其中 $P(y=1|x)$ 是给定输入 $x$ 时，输出为 1 的概率。

**5. 似然函数**

为了估计模型参数，我们使用极大似然估计法。对于给定的训练数据集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$，似然函数为：

$$
L(\beta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\beta) = \prod_{i=1}^{m} [\sigma(z^{(i)})]^{y^{(i)}} [1 - \sigma(z^{(i)})]^{1-y^{(i)}}
$$

其中 $z^{(i)} = \beta_0 + \beta_1 x_1^{(i)} + \cdots + \beta_n x_n^{(i)}$。

**6. 对数似然函数**

为了简化计算，我们通常使用对数似然函数：

$$
\ell(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
$$

**7. 参数估计**

通过最大化对数似然函数 $\ell(\beta)$，我们可以找到最优的参数 $\beta$。这通常通过梯度下降或其他优化算法来实现。

**8. 交叉熵损失函数**

在逻辑回归中，我们使用交叉熵（Cross Entropy）作为损失函数。对于二分类问题，交叉熵损失函数定义如下：

$$
L(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

其中：
- $y$ 是真实标签（0或1）
- $\hat{y}$ 是模型预测的概率，即 $\sigma(z)$
- $z = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n$

对于整个数据集，损失函数为：

$$
J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]
$$

**9. 梯度推导**

为了使用梯度下降最小化损失函数，我们需要计算损失函数对参数 $\beta_j$ 的偏导数：

1) 首先，将 $\hat{y} = \sigma(z)$ 代入损失函数：

$$
J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\sigma(z^{(i)})) + (1-y^{(i)})\log(1-\sigma(z^{(i)}))]
$$

2) 对 $\beta_j$ 求偏导：

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\frac{\partial}{\partial \beta_j}\log(\sigma(z^{(i)})) + (1-y^{(i)})\frac{\partial}{\partial \beta_j}\log(1-\sigma(z^{(i)}))]
$$

3) 利用链式法则：

$$
\frac{\partial}{\partial \beta_j}\log(\sigma(z)) = \frac{1}{\sigma(z)}\sigma(z)(1-\sigma(z))x_j
$$

$$
\frac{\partial}{\partial \beta_j}\log(1-\sigma(z)) = -\frac{\sigma(z)}{1-\sigma(z)}(1-\sigma(z))x_j = -\sigma(z)x_j
$$

4) 代入并化简：

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}(1-\sigma(z^{(i)}))x_j^{(i)} - (1-y^{(i)})\sigma(z^{(i)})x_j^{(i)}]
$$

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \sigma(z^{(i)}))x_j^{(i)}
$$

5) 梯度下降更新规则：

$$
\beta_j := \beta_j - \alpha\frac{\partial J}{\partial \beta_j} = \beta_j + \frac{\alpha}{m}\sum_{i=1}^{m}(y^{(i)} - \sigma(z^{(i)}))x_j^{(i)}
$$

其中 $\alpha$ 是学习率。

**10. 预测**

一旦模型参数 $\beta$ 被估计，我们可以使用逻辑回归模型进行预测。对于给定的输入 $x$，计算 $P(y=1|x)$，并根据阈值（通常为0.5）进行分类。


```python
import numpy as np

# 定义sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 梯度更新函数
def gradient_descent(X, y, beta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        z = np.dot(X, beta)
        predictions = sigmoid(z)
        errors = y - predictions
        gradient = np.dot(X.T, errors)
        beta += alpha * gradient
    return beta

# 生成带有随机性的数据
np.random.seed(42)  # 设置随机种子以确保结果可重复
X = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100)])  # 100个样本，每个样本3个特征，特征带有随机性
y = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X])  # 根据特征和生成标签
beta = np.zeros(X.shape[1])  # 初始化参数为0
alpha = 0.01  # 学习率
iterations = 1000  # 迭代次数

# 使用梯度下降拟合逻辑回归参数
beta = gradient_descent(X, y, beta, alpha, iterations)
print("拟合的参数:", beta)

# 生成带有随机性的测试数据
X_test = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100, 120)])  # 20个测试样本，每个样本3个特征
y_test = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X_test])  # 根据特征和生成标签

# 预测函数
def predict(X, beta):
    return sigmoid(np.dot(X, beta)) >= 0.5

# 计算模型准确率
predictions = predict(X_test, beta)
accuracy = np.mean(predictions == y_test)
print("模型准确率:", accuracy)
```
```
拟合的参数: [ 249.40764222  -13.4209779  -168.15650401]
模型准确率: 1.0
```

#### 参考文献
1. https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
2. https://blog.csdn.net/weixin_60737527/article/details/124141293


