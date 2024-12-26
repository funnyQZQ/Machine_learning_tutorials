### Likelihood Function

**1. Bernoulli Distribution**

For each sample $i$, the output $y^{(i)}$ is a binary variable (0 or 1), with the probability distribution as follows:
If $y^{(i)} = 1$, then $P(y^{(i)}|x^{(i)};\beta) = \sigma(z^{(i)})$
If $y^{(i)} = 0$, then $P(y^{(i)}|x^{(i)};\beta) = 1 - \sigma(z^{(i)})$

**2. Joint Probability**

For the entire dataset $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, assuming the samples are independently and identically distributed, the joint probability (i.e., the likelihood function) is:
$$
L(\beta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\beta)
$$
Combining with the Bernoulli distribution: the probability for each sample is combined to get:
$$
L(\beta) = \prod_{i=1}^{m} [\sigma(z^{(i)})]^{y^{(i)}} [1 - \sigma(z^{(i)})]^{1-y^{(i)}}
$$
Here, $[\sigma(z^{(i)})]^{y^{(i)}}$ indicates using $\sigma(z^{(i)})$ when $y^{(i)} = 1$, and $[1 - \sigma(z^{(i)})]^{1-y^{(i)}}$ indicates using $1 - \sigma(z^{(i)})$ when $y^{(i)} = 0$.

**3. Log-Likelihood Function**

To simplify calculations, we often use the log-likelihood function. The log-likelihood function is the logarithm of the likelihood function:
$$
\ell(\beta) = \log L(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
$$
The advantage of the log-likelihood function is that it converts products into sums, making calculations more straightforward and aiding numerical stability.