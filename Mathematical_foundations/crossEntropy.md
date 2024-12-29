### Cross Entropy

Cross entropy formula can be understood from two perspectives: information theory and maximum likelihood estimation.

**1. Information Theory Perspective**

In information theory, entropy is defined as:

$$
H(p) = -\sum_x p(x)\log(p(x))
$$

Cross entropy is defined as:

$$
H(p,q) = -\sum_{i=1}^{K} p(x_i)\log(q(x_i))
$$

Where:
- $p(x_i)$ is the probability of the true distribution for class $i$
- $q(x_i)$ is the probability of the predicted distribution for class $i$
- $K$ is the total number of classes
- For classification problems, $p(x_i)$ is typically one-hot encoded (1 for the true class, 0 for others)

**2. Maximum Likelihood Perspective**

From a statistical perspective, for binary classification, the likelihood function is:

$$
L(\theta) = \prod_{i=1}^{m} (\hat{y}^{(i)})^{y^{(i)}} (1-\hat{y}^{(i)})^{1-y^{(i)}}
$$

Taking the logarithm gives us the log-likelihood function:

$$
\ln L(\theta) = \sum_{i=1}^{m} [y^{(i)}\ln(\hat{y}^{(i)}) + (1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

To maximize likelihood, we typically take the negative and minimize:

$$
-\ln L(\theta) = -\sum_{i=1}^{m} [y^{(i)}\ln(\hat{y}^{(i)}) + (1-y^{(i)})\ln(1-\hat{y}^{(i)})]
$$

This is identical to the cross entropy form.

**3. Applications of Cross Entropy**

For **binary classification**, the basic cross entropy formula is:

$$
H(y,\hat{y}) = -[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]
$$

Where:
- $y$ is the true label (0 or 1)
- $\hat{y}$ is the predicted probability
- $\log$ is typically the natural logarithm $\ln$

For **multi-class classification (K classes)**, the cross entropy formula extends to:

$$
H(y,\hat{y}) = -\sum_{i=1}^{K} y_i\log(\hat{y_i})
$$

Where:
- $y_i$ is the one-hot encoded true label (1 for the correct class, 0 for others)
- $\hat{y_i}$ is the predicted probability for class i
- $K$ is the total number of classes

For **batch data**, the cross entropy loss function is:

$$
J = -\frac{1}{m}\sum_{j=1}^{m}\sum_{i=1}^{K} y_i^{(j)}\log(\hat{y_i}^{(j)})
$$

Where:
- $m$ is the number of samples
- $y_i^{(j)}$ is the true label for class i of sample j
- $\hat{y_i}^{(j)}$ is the predicted probability for class i of sample j

**Key Properties**:
1. When the true label is 1, only the $\log(\hat{y})$ term matters
2. When the true label is 0, only the $\log(1-\hat{y})$ term matters
3. Cross entropy is always non-negative
4. The more accurate the prediction, the smaller the cross entropy loss
5. When prediction is perfect ($\hat{y}=y$), the loss is 0

**4. Code Implementation**

**NumPy Implementation**
```python
import numpy as np

def cross_entropy_numpy(y_true, y_pred, epsilon=1e-15):
    """
    Calculate cross entropy loss
    
    Parameters:
        y_true: true labels (one-hot encoded)
        y_pred: predicted probabilities
        epsilon: small value to prevent log(0)
    """
    # Clip prediction values to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Binary classification case
    if y_true.ndim == 1:
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Multi-class classification case
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Binary classification
y_true_binary = np.array([1, 0, 1, 1])
y_pred_binary = np.array([0.9, 0.1, 0.8, 0.7])
loss_binary = cross_entropy_numpy(y_true_binary, y_pred_binary)
print(f"Binary Cross Entropy Loss: {loss_binary:.4f}")

# Multi-class classification
y_true_multi = np.array([[1,0,0], [0,1,0], [0,0,1]])  # 3 samples, 3 classes
y_pred_multi = np.array([[0.8,0.1,0.1], [0.2,0.7,0.1], [0.1,0.2,0.7]])
loss_multi = cross_entropy_numpy(y_true_multi, y_pred_multi)
print(f"Multi-class Cross Entropy Loss: {loss_multi:.4f}")
```

Output:
```
Binary Cross Entropy Loss: 0.1976
Multi-class Cross Entropy Loss: 0.3122
```