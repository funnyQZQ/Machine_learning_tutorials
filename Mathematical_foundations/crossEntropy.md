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