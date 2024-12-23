### Linear Regression

#### 1. Model Assumptions
Linear regression assumes a linear relationship between the independent variable $x$ and the dependent variable $y$. The model is represented as:

$$ 
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon 
$$

Where:
- $y$ is the dependent variable (target variable).
- $x_1, x_2, \ldots, x_n$ are the independent variables (features).
- $\beta_0$ is the intercept.
- $\beta_1, \beta_2, \ldots, \beta_n$ are the regression coefficients.
- $\epsilon$ is the error term, assumed to have a mean of zero and normally distributed.


#### 2. Objective
The objective of linear regression is to find the optimal regression coefficients $\beta$ that minimize the sum of squared differences between the predicted and actual values. This process is known as Ordinary Least Squares (OLS).

#### 3. Solving the problem

#### 3.1 Analytical solution

We have a linear regression model:
$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon
$$
Our goal is to find the parameters $\beta = [\beta_0, \beta_1, \cdots, \beta_n]$ that minimize the sum of squared errors between the predicted and actual values.
$$
\beta = [\beta_0, \beta_1, \cdots, \beta_n]
$$
that minimize the sum of squared errors between the predicted and actual values.

The problem can be expressed in matrix form as:
$$
\mathbf{y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon}
$$
Where: 
- $\mathbf{y}$ is an $m \times 1$ vector of the target variable.
- $\mathbf{X}$ is an $m \times (n+1)$ matrix, with the first column being all ones (for the intercept $\beta_0$), and the remaining columns are the feature values.
- $\boldsymbol{\beta}$ is an $(n+1) \times 1$ vector of regression coefficients.
- $\boldsymbol{\epsilon}$ is an $m \times 1$ vector of errors.

**Objective Function**  

We aim to minimize the sum of squared errors:  
The original objective function can be defined as:
$$
S = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
where $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.

$$
S = (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})
$$
Expanding the expression, we get:
$$
S = (\mathbf{y}^T - \boldsymbol{\beta}^T \mathbf{X}^T)(\mathbf{y} - \mathbf{X} \boldsymbol{\beta})
$$
$$
S = \mathbf{y}^T \mathbf{y} - \mathbf{y}^T \mathbf{X} \boldsymbol{\beta} - \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$
Since $\mathbf{y}^T \mathbf{X} \boldsymbol{\beta}$ and $\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{y}$ are scalars and equal, we can simplify to:
$$
S = \mathbf{y}^T \mathbf{y} - 2\mathbf{y}^T \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$


**Derivation Steps**
1. Expand the Objective Function:
$$
S = \mathbf{y}^T \mathbf{y} - 2\mathbf{y}^T \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$
2. Take the Derivative with Respect to $\boldsymbol{\beta}$ and Set it to Zero:

To find the optimal $\boldsymbol{\beta}$, we need to take the derivative of the objective function $S$ with respect to $\boldsymbol{\beta}$ and set it to zero. The derivative of $S$ with respect to $\boldsymbol{\beta}$ is given by: 
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = \frac{\partial}{\partial \boldsymbol{\beta}} \left( \mathbf{y}^T \mathbf{y} - 2\mathbf{y}^T \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} \right)
$$

- Since $\mathbf{y}^T \mathbf{y}$ is a constant with respect to $\boldsymbol{\beta}$, its derivative is zero. 

- The derivative of $-2\mathbf{y}^T \mathbf{X} \boldsymbol{\beta}$ with respect to $\boldsymbol{\beta}$ is $-2\mathbf{X}^T \mathbf{y}$ because $\mathbf{y}^T \mathbf{X}$ is a constant matrix and the derivative of a linear term $\mathbf{X} \boldsymbol{\beta}$ with respect to $\boldsymbol{\beta}$ is $\mathbf{X}$. Therefore, the scalar factor $-2$ remains, resulting in $-2\mathbf{X}^T \mathbf{y}$. 

- Derivative of a Quadratic Form: For a quadratic form $\boldsymbol{\beta}^T A \boldsymbol{\beta}$, the derivative with respect to $\boldsymbol{\beta}$ is given by:

$$
\frac{\partial}{\partial \boldsymbol{\beta}} (\boldsymbol{\beta}^T A \boldsymbol{\beta}) = (A + A^T) \boldsymbol{\beta}
$$

Since $A = \mathbf{X}^T \mathbf{X}$ is symmetric ($A = A^T$), the derivative simplifies to:

$$
2A\boldsymbol{\beta} = 2\mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$

Therefore, we have:

$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
$$
Setting the derivative to zero, we get:
$$
-2\mathbf{X}^T \mathbf{y} + 2\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = 0
$$
3. Solve for $\boldsymbol{\beta}$:
$$
\mathbf{X}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{X}^T \mathbf{y}
$$
Assuming $\mathbf{X}^T \mathbf{X}$ is invertible, we can solve for $\boldsymbol{\beta}$:
 X is invertible, we can solve for 
$$
\boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

```python
import numpy as np

def linear_regression_weights(X, y):
    """
    Calculate the analytical solution for the weights of a linear regression model.
    """
    # Calculate the transpose of X
    X_transpose = np.transpose(X)
    
    # Calculate the dot product of X_transpose and X
    X_transpose_X = np.dot(X_transpose, X)
    
    # Calculate the inverse of X_transpose_X
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    
    # Calculate the dot product of X_transpose and y
    X_transpose_y = np.dot(X_transpose, y)
    
    # Calculate the weights
    weights = np.dot(X_transpose_X_inv, X_transpose_y)
    
    return weights

# usage
X = np.array([[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]])
y = np.array([[1], [2], [3], [4]])
weights = linear_regression_weights(X, y)
print(weights)
```
    [[-1.42108547e-14]
    [ 1.00000000e+00]
    [ 7.10542736e-15]]

#### 3.2 Gradient Descent

Gradient Descent is an iterative optimization algorithm used to minimize the cost function. For linear regression, the cost function is defined as:
$$
S = \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2
$$
where $m$ is the number of training examples, $y_i$ is the actual value, and $\hat{y}_i$ is the predicted value.

The goal is to find the weights $\boldsymbol{\beta}$ that minimize the cost function $S$. The gradient descent algorithm updates the weights iteratively using the following rule:
$$
\boldsymbol{\beta} := \boldsymbol{\beta} - \alpha \frac{\partial S}{\partial \boldsymbol{\beta}}
$$
where $\alpha$ is the learning rate.

To derive the update rule, we first need to compute the gradient of the cost function with respect to the weights $\boldsymbol{\beta}$:
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = \frac{\partial}{\partial \boldsymbol{\beta}} \left( \frac{1}{2m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2 \right)
$$

Expanding $\hat{y}_i$ as $\mathbf{X} \boldsymbol{\beta}$:
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = \frac{\partial}{\partial \boldsymbol{\beta}} \left( \frac{1}{2m} \sum_{i=1}^{m} (y_i - \mathbf{X}_i \boldsymbol{\beta})^2 \right)
$$

Using the chain rule, we get:
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \mathbf{X}_i \boldsymbol{\beta})(- \mathbf{X}_i)
$$

Simplifying, we obtain:
$$
\frac{\partial S}{\partial \boldsymbol{\beta}} = -\frac{1}{m} \mathbf{X}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})
$$

Thus, the gradient descent update rule becomes:
$$
\boldsymbol{\beta} := \boldsymbol{\beta} + \alpha \frac{1}{m} \mathbf{X}^T (\mathbf{y} - \mathbf{X} \boldsymbol{\beta})
$$

This update rule is applied iteratively until the cost function converges to a minimum value.

```python
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
```
    [[6.94127936e-05]
    [9.99936851e-01]
    [1.19586769e-05]]


#### References
- https://zh.d2l.ai/chapter_linear-networks/linear-regression.html

