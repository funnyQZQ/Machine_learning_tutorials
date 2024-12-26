### 特征值和特征向量

**特征向量**：这些是在线性变换应用时仅通过一个标量因子改变的非零向量。数学上，对于一个矩阵 $A$，特征向量 $v$ 满足：

$$
A\vec{v}=\lambda\vec{v}
$$

其中 $\lambda$ 是对应于特征向量 $\vec{v}$ 的特征值。  
*注意*：只有方阵才有特征值和特征向量。非方阵没有特征值或特征向量。

**特征值**：这些是表示特征向量在变换过程中被拉伸或压缩程度的标量。

矩阵 $A$ 的特征向量有**两个重要性质**：

1. 如果 $X_1$ 和 $X_2$ 是对应于特征值 $\lambda$ 的 $A$ 的特征向量，那么 $X_1 + X_2$ 也是对应于特征值 $\lambda$ 的 $A$ 的特征向量。

2. 如果 $X$ 是对应于特征值 $\lambda$ 的 $A$ 的特征向量，并且 $k$ 是一个非零常数，那么 $kX$ 也是对应于特征值 $\lambda$ 的 $A$ 的特征向量。

**性质1的例子**

考虑矩阵   

$$
A = \begin{bmatrix} 
2 & 0 \\ 
0 & 3 
\end{bmatrix}
$$ 

特征值 $\lambda = 2$。对应于 $\lambda = 2$ 的特征向量形式为 $X = \begin{bmatrix} x \\ 0 \end{bmatrix}$。

令 $X_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ 和 $X_2 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$。两者都是 $A$ 的特征向量，对应于 $\lambda = 2$。

它们的和 $X_1 + X_2 = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$ 也是 $A$ 的特征向量，对应于 $\lambda = 2$。

**性质2的例子**

使用相同的矩阵 $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ 和特征值 $\lambda = 2$，考虑特征向量 $X = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$。

如果 $k = 3$，那么 $kX = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$ 也是 $A$ 的特征向量，对应于 $\lambda = 2$。

#### 如何计算方阵的特征值和特征向量？

#### 示例矩阵

考虑矩阵 $A$：

$$ 
A = \begin{bmatrix} 
4 & 1 & 2 \\ 
0 & 3 & -1 \\ 
0 & 0 & 2 
\end{bmatrix} 
$$

#### 步骤1：计算特征值

1. **特征方程**：计算 $\det(A - \lambda I) = 0$，其中 $I$ 是单位矩阵，$\lambda$ 是特征值。

2. **计算行列式**：

   $$
   \det(A - \lambda I) = \det \begin{bmatrix} 
   4-\lambda & 1 & 2 \\ 
   0 & 3-\lambda & -1 \\ 
   0 & 0 & 2-\lambda 
   \end{bmatrix} 
   $$

   由于这是一个上三角矩阵，行列式是对角线元素的乘积：

   $$
   (4-\lambda)(3-\lambda)(2-\lambda) = 0
   $$

3. **求解特征值**：解方程得到 $\lambda_1 = 4$，$\lambda_2 = 3$，$\lambda_3 = 2$。

#### 步骤2：计算特征向量

对于每个特征值，解 $(A - \lambda I)X = 0$：

1. **特征值 $\lambda_1 = 4$**：

   $$
   (A - 4I) = \begin{bmatrix} 
   0 & 1 & 2 \\ 
   0 & -1 & -1 \\ 
   0 & 0 & -2 
   \end{bmatrix} 
   $$

   解 $(A - 4I)X = 0$ 得到特征向量 $X_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$。

2. **特征值 $\lambda_2 = 3$**：

   $$
   (A - 3I) = \begin{bmatrix} 
   1 & 1 & 2 \\ 
   0 & 0 & -1 \\ 
   0 & 0 & -1 
   \end{bmatrix} 
   $$

   解 $(A - 3I)X = 0$ 得到特征向量 $X_2 = \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}$。

3. **特征值 $\lambda_3 = 2$**：

   $$
   (A - 2I) = \begin{bmatrix} 
   2 & 1 & 2 \\ 
   0 & 1 & -1 \\ 
   0 & 0 & 0 
   \end{bmatrix} 
   $$

   解 $(A - 2I)X = 0$ 得到特征向量 $X_3 = \begin{bmatrix} -3 \\ 2 \\ 2 \end{bmatrix}$。

```python
import numpy as np

# 定义矩阵 A
A = np.array([
    [4, 1, 2],
    [0, 3, -1],
    [0, 0, 2]
])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 显示结果
print("特征值:\n", eigenvalues)

print("\n特征向量:\n", eigenvectors)
```

    特征值:
     [4. 3. 2.]
    
    特征向量:
     [[ 1.         -0.70710678 -0.72760688]
     [ 0.          0.70710678  0.48507125]
     [ 0.          0.          0.48507125]]
    

#### 参考文献
1. https://byjus.com/maths/eigen-values/
2. https://zhuanlan.zhihu.com/p/165382601