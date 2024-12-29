### 交叉熵(Cross Entropy)


交叉熵公式可以从两个角度来理解：信息论角度和最大似然估计角度。

**1. 信息论角度**

在信息论中，信息熵定义为：

$$
H(p) = -\sum_x p(x)\log(p(x))
$$

交叉熵则定义为：

$$
H(p,q) = -\sum_{i=1}^{K} p(x_i)\log(q(x_i))
$$

其中：
- $p(x_i)$ 是真实分布在第 $i$ 类的概率
- $q(x_i)$ 是预测分布在第 $i$ 类的概率
- $K$ 是类别总数
- 对于分类问题，$p(x_i)$ 通常是one-hot编码（只有真实类别处为1，其他为0）

**2. 最大似然估计角度**

从统计学角度，对于二分类问题，似然函数为：

$$
L(\theta) = \prod_{i=1}^{m} (\hat{y}^{(i)})^{y^{(i)}} (1-\hat{y}^{(i)})^{1-y^{(i)}}
$$

取对数得到对数似然函数：

$$
\ln L(\theta) = \sum_{i=1}^{m} [y^{(i)}\ln(\hat{y}^{(i)}) + (1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

为了最大化似然，通常取负值并最小化，得到：

$$
-\ln L(\theta) = -\sum_{i=1}^{m} [y^{(i)}\ln(\hat{y}^{(i)}) + (1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

这与交叉熵的形式完全一致。

**3. 交叉熵的应用**

对于**二分类问题**，交叉熵的基本公式是：

$$
H(y,\hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

其中：
- $y$ 是真实标签（0或1）
- $\hat{y}$ 是预测概率
- $\log$ 通常是自然对数 $\ln$

对于**多分类问题（K类）**，交叉熵公式扩展为：

$$
H(y,\hat{y}) = -\sum_{i=1}^{K} y_i\log(\hat{y_i})
$$

其中：
- $y_i$ 是真实标签的one-hot编码（只有一个位置是1，其他都是0）
- $\hat{y_i}$ 是第i类的预测概率
- $K$ 是类别总数

对于**批量数据（batch）**，交叉熵损失函数为：

$$
J = -\frac{1}{m}\sum_{j=1}^{m}\sum_{i=1}^{K} y_i^{(j)}\log(\hat{y_i}^{(j)})
$$

其中：
- $m$ 是样本数量
- $y_i^{(j)}$ 是第j个样本对第i类的真实标签
- $\hat{y_i}^{(j)}$ 是第j个样本对第i类的预测概率

**特点说明**：
1. 当真实标签为1时，只关注 $\log(\hat{y})$ 项
2. 当真实标签为0时，只关注 $\log(1-\hat{y})$ 项
3. 交叉熵总是非负的
4. 预测越准确，交叉熵损失越小
5. 预测完全正确时（$\hat{y}=y$），损失为0


**4. 代码实现**

**NumPy实现**
```python
import numpy as np

def cross_entropy_numpy(y_true, y_pred, epsilon=1e-15):
    """
    计算交叉熵损失
    
    参数:
        y_true: 真实标签 (one-hot编码)
        y_pred: 预测概率
        epsilon: 小数值，防止log(0)
    """
    # 裁剪预测值，避免log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # 二分类情况
    if y_true.ndim == 1:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # 多分类情况
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# 示例
# 二分类
y_true_binary = np.array([1, 0, 1, 1])
y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7])
loss_binary = cross_entropy_numpy(y_true_binary, y_pred_binary)
print(f"Binary Cross Entropy Loss: {loss_binary:.4f}")

# 多分类
y_true_multi = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 3个样本，3个类别
y_pred_multi = np.array([[0.8,0.1,0.1], [0.2,0.7,0.1], [0.1,0.2,0.7]])
loss_multi = cross_entropy_numpy(y_true_multi, y_pred_multi)
print(f"Multi-class Cross Entropy Loss: {loss_multi:.4f}")
```

输出结果：
```
Binary Cross Entropy Loss: 0.1976
Multi-class Cross Entropy Loss: 0.3122
```
