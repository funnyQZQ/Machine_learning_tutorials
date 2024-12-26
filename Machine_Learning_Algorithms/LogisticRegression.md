### Logistic Regression

Logistic regression is a statistical method used for classification problems. Despite having "regression" in its name, it's primarily used for binary classification problems. Here's the derivation process:

**1. Basic Concepts of Logistic Regression**

The goal of logistic regression is to predict a binary dependent variable (0 or 1) given one or more independent variables. This is achieved by estimating the probability of an event occurring.

**2. Linear Model**

First, assume we have a linear model:

$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

where $z$ is the linear combination, $\beta_0, \beta_1, \ldots, \beta_n$ are model parameters, and $x_1, x_2, \ldots, x_n$ are independent variables.

**3. Sigmoid Function**

To convert the linear model's output into a probability, we use the Sigmoid function (also known as the logistic function):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The Sigmoid function maps any real number to the interval (0, 1), making it suitable for representing probabilities.

**4. Logistic Regression Model**

The logistic regression model can be represented as:

$$
P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
$$

where $P(y=1|x)$ is the probability of output being 1 given input $x$.

**5. Likelihood Function**

To estimate model parameters, we use Maximum Likelihood Estimation. For a given training dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, the likelihood function is:

$$
L(\beta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\beta) = \prod_{i=1}^{m} [\sigma(z^{(i)})]^{y^{(i)}} [1 - \sigma(z^{(i)})]^{1-y^{(i)}}
$$

where $z^{(i)} = \beta_0 + \beta_1 x_1^{(i)} + \cdots + \beta_n x_n^{(i)}$.

**6. Log-Likelihood Function**

To simplify calculations, we typically use the log-likelihood function:

$$
\ell(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
$$

**7. Parameter Estimation**

By maximizing the log-likelihood function $\ell(\beta)$, we can find the optimal parameters $\beta$. This is typically achieved through gradient descent or other optimization algorithms.

**8. Cross-Entropy Loss Function**

In logistic regression, we use Cross-Entropy as the loss function. For binary classification, the cross-entropy loss function is defined as:

$$
L(y, \hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

where:
- $y$ is the true label (0 or 1)
- $\hat{y}$ is the predicted probability, i.e., $\sigma(z)$
- $z = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n$

For the entire dataset, the loss function is:

$$
J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]
$$

**9. Gradient Derivation**

To minimize the loss function using gradient descent, we need to calculate the partial derivatives of the loss function with respect to parameters $\beta_j$:

1) First, substitute $\hat{y} = \sigma(z)$ into the loss function:

$$
J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\log(\sigma(z^{(i)})) + (1-y^{(i)})\log(1-\sigma(z^{(i)}))]
$$

2) Take partial derivative with respect to $\beta_j$:

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}\frac{\partial}{\partial \beta_j}\log(\sigma(z^{(i)})) + (1-y^{(i)})\frac{\partial}{\partial \beta_j}\log(1-\sigma(z^{(i)}))]
$$

3) Using the chain rule:

$$
\frac{\partial}{\partial \beta_j}\log(\sigma(z)) = \frac{1}{\sigma(z)}\sigma(z)(1-\sigma(z))x_j
$$

$$
\frac{\partial}{\partial \beta_j}\log(1-\sigma(z)) = -\frac{\sigma(z)}{1-\sigma(z)}(1-\sigma(z))x_j = -\sigma(z)x_j
$$

4) Substitute and simplify:

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}(1-\sigma(z^{(i)}))x_j^{(i)} - (1-y^{(i)})\sigma(z^{(i)})x_j^{(i)}]
$$

$$
\frac{\partial J}{\partial \beta_j} = -\frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \sigma(z^{(i)}))x_j^{(i)}
$$

5) Gradient descent update rule:

$$
\beta_j := \beta_j - \alpha\frac{\partial J}{\partial \beta_j} = \beta_j + \frac{\alpha}{m}\sum_{i=1}^{m}(y^{(i)} - \sigma(z^{(i)}))x_j^{(i)}
$$

where $\alpha$ is the learning rate.

**10. Prediction**

Once the model parameters $\beta$ are estimated, we can use the logistic regression model for predictions. For a given input $x$, calculate $P(y=1|x)$ and classify based on a threshold (typically 0.5).


```python
import numpy as np

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent function
def gradient_descent(X, y, beta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        z = np.dot(X, beta)
        predictions = sigmoid(z)
        errors = y - predictions
        gradient = np.dot(X.T, errors)
        beta += alpha * gradient
    return beta

# Generate data with randomness
np.random.seed(42)  # Set random seed for reproducibility
X = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100)])  # 100 samples, 3 features each with randomness
y = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X])  # Generate labels based on feature sum
beta = np.zeros(X.shape[1])  # Initialize parameters to zero
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Fit logistic regression parameters using gradient descent
beta = gradient_descent(X, y, beta, alpha, iterations)
print("Fitted parameters:", beta)

# Generate test data with randomness
X_test = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100, 120)])  # 20 test samples
y_test = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X_test])  # Generate test labels

# Prediction function
def predict(X, beta):
    return sigmoid(np.dot(X, beta)) >= 0.5

# Calculate model accuracy
predictions = predict(X_test, beta)
accuracy = np.mean(predictions == y_test)
print("Model accuracy:", accuracy)
```
```
Fitted parameters: [ 249.40764222  -13.4209779  -168.15650401]
Model accuracy: 1.0
```

#### References
1. https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
2. https://blog.csdn.net/weixin_60737527/article/details/124141293