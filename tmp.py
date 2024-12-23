import numpy as np

def gradient_descent(X, y, alpha=0.01, iterations=1000):
    """
    use gradient descent to fit the linear regression model
    """
    m, n = X.shape
    beta = np.zeros((n, 1))
    
    for _ in range(iterations):
        gradient = (1/m) * np.dot(X.T, y - np.dot(X, beta))
        beta = beta + alpha * gradient
    
    return beta

# usage
X = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]])
y = np.array([[1], [2], [3], [4]])
alpha = 0.005
iterations = 100000
beta = gradient_descent(X, y, alpha, iterations)
print(beta)
