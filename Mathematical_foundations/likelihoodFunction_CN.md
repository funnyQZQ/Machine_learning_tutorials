### 似然函数
**1. 伯努利分布**

对于每个样本 $i$，输出 $y^{(i)}$ 是一个二元变量（0或1），其概率分布为：
如果 $y^{(i)} = 1$，则 $P(y^{(i)}|x^{(i)};\beta) = \sigma(z^{(i)})$
如果 $y^{(i)} = 0$，则 $P(y^{(i)}|x^{(i)};\beta) = 1 - \sigma(z^{(i)})$

**2. 联合概率**

对于整个数据集 $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$，假设样本是独立同分布的，联合概率（即似然函数）为：
$$
L(\beta) = \prod_{i=1}^{m} P(y^{(i)}|x^{(i)};\beta)
$$
结合伯努利分布：将每个样本的概率结合起来，得到：
$$
L(\beta) = \prod_{i=1}^{m} [\sigma(z^{(i)})]^{y^{(i)}} [1 - \sigma(z^{(i)})]^{1-y^{(i)}}
$$
这里，$[\sigma(z^{(i)})]^{y^{(i)}}$ 表示当 $y^{(i)} = 1$ 时，使用 $\sigma(z^{(i)})$，而 $[1 - \sigma(z^{(i)})]^{1-y^{(i)}}$ 表示当 $y^{(i)} = 0$ 时，使用 $1 - \sigma(z^{(i)})$。

**3. 对数似然函数**

为了简化计算，我们通常使用对数似然函数。对数似然函数是似然函数取对数后的形式：
$$
\ell(\beta) = \log L(\beta) = \sum_{i=1}^{m} \left[ y^{(i)} \log(\sigma(z^{(i)})) + (1 - y^{(i)}) \log(1 - \sigma(z^{(i)})) \right]
$$
对数似然函数的优点在于将乘积转换为求和，使得计算更加简便，同时也有助于数值稳定性。
