### Logistic Regression

Logistic regression is a statistical method used for classification problems. Despite the term "regression" in its name, it is primarily used for binary classification problems. Below is the derivation process of logistic regression:

**1. Basic Concept of Logistic Regression**

The goal of logistic regression is to predict a binary dependent variable (0 or 1), given one or more independent variables. It achieves this by estimating the probability of an event occurring.

**2. Linear Model**

First, assume we have a linear model:

$$
z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n
$$

where $z$ is a linear combination, $\beta_0, \beta_1, \ldots, \beta_n$ are model parameters, and $x_1, x_2, \ldots, x_n$ are independent variables.

**3. Sigmoid Function**

To convert the output of the linear model into a probability, we use the Sigmoid function (also known as the logistic function):

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The Sigmoid function maps any real number to the interval (0, 1), making it suitable for representing probabilities.

**4. Logistic Regression Model**

The logistic regression model can be expressed as:

$$
P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n)}}
$$

where $P(y=1|x)$ is the probability of the output being 1 given the input $x$.

**5. Likelihood Function**

To estimate the model parameters, we use the maximum likelihood estimation method. For a given training dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, the likelihood function is:

$$
L(\beta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\beta) = \prod_{i=1}^{m} [\sigma(z^{(i)})]^{y^{(i)}} [1 - \sigma(z^{(i)})]^{1-y^{(i)}}
$$

where $z^{(i)} = \beta_0 + \beta_1 x_1^{(i)} + \cdots + \beta_n x_n^{(i)}$.

**6. Log-Likelihood Function**

To simplify calculations, we often use the log-likelihood function:

$$
\ell(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
$$

**7. Parameter Estimation**

By maximizing the log-likelihood function $\ell(\beta)$, we can find the optimal parameters $\beta$. This is usually achieved through gradient descent or other optimization algorithms.

**8. Gradient Descent**

To derive the gradient of the log-likelihood function, first write the log-likelihood function:

$$
\ell(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)} ) \log(1 - \sigma(z^{(i)})) \right]
$$

Differentiate with respect to $\beta_j$:

$$
\frac{\partial \ell(\beta)}{\partial \beta_j} = \sum_{i=1}^{m} \left[ y^{(i)} \frac{\partial \log(\sigma(z^{(i)}))}{\partial \beta_j} + (1 - y^{(i)}) \frac{\partial \log(1 - \sigma(z^{(i)}))}{\partial \beta_j} \right]
$$

Using the chain rule, differentiate:

$$
\frac{\partial \log(\sigma(z^{(i)}))}{\partial \beta_j} = \frac{1}{\sigma(z^{(i)})} \cdot \sigma(z^{(i)}) \cdot (1 - \sigma(z^{(i)})) \cdot x_j^{(i)} = (1 - \sigma(z^{(i)})) x_j^{(i)}
$$

$$
\frac{\partial \log(1 - \sigma(z^{(i)}))}{\partial \beta_j} = \frac{1}{1 - \sigma(z^{(i)})} \cdot (-\sigma(z^{(i)})) \cdot (1 - \sigma(z^{(i)})) \cdot x_j^{(i)} = -\sigma(z^{(i)}) x_j^{(i)}
$$

Substitute the above results into the gradient formula:

$$
\frac{\partial \ell(\beta)}{\partial \beta_j} = \sum_{i=1}^{m} \left[ y^{(i)} (1 - \sigma(z^{(i)})) x_j^{(i)} - (1 - y^{(i)}) \sigma(z^{(i)}) x_j^{(i)} \right]
$$

Simplify to get:

$$
\frac{\partial \ell(\beta)}{\partial \beta_j} = \sum_{i=1}^{m} (y^{(i)} - \sigma(z^{(i)})) x_j^{(i)}
$$

Update parameters using gradient descent:

$$
\beta_j := \beta_j + \alpha \sum_{i=1}^{m} (y^{(i)} - \sigma(z^{(i)})) x_j^{(i)}
$$

where $\alpha$ is the learning rate.

**9. Prediction**

Once the model parameters $\beta$ are estimated, we can use the logistic regression model for prediction. For a given input $x$, compute $P(y=1|x)$ and classify based on a threshold (usually 0.5).

```python
import numpy as np

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient update function
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
X = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100)])  # 100 samples, each with 3 features, features with randomness
y = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X])  # Generate labels based on feature sum
beta = np.zeros(X.shape[1])  # Initialize parameters to zero
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations

# Fit logistic regression parameters using gradient descent
beta = gradient_descent(X, y, beta, alpha, iterations)
print("Fitted parameters:", beta)

# Generate test data with randomness
X_test = np.array([[i + np.random.normal(0, 1), i + 1 + np.random.normal(0, 1), i + 2 + np.random.normal(0, 1)] for i in range(100, 120)])  # 20 test samples, each with 3 features
y_test = np.array([1 if (x[0] + x[1] + x[2]) > 150 else 0 for x in X_test])  # Generate labels based on feature sum

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