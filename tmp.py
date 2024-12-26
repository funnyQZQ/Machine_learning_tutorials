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