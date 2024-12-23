### Eigenvalues and Eigenvectors

**Eigenvectors**: These are non-zero vectors that only change by a scalar factor when a linear transformation is applied. Mathematically, for a matrix $A$, an eigenvector $v$ satisfies:
$$A\vec{v}=\lambda\vec{v}$$
where $\lambda$ is the eigenvalue corresponding to the eigenvector $\vec{v}$.  
*Note*: Only square matrices have eigenvalues and eigenvectors. Non-square matrices do not have eigenvalues or eigenvectors.



**Eigenvalues**: These are scalars that represent how much the eigenvector is stretched or compressed during the transformation.

The eigenvectors of matrix $ A $ have **two important properties**:

1. If $X_1$ and $X_2$ are eigenvectors of $A$ corresponding to the eigenvalue $\lambda$, then $X_1 + X_2$ is also an eigenvector of $A$ corresponding to the eigenvalue $\lambda$.


2. If $X$ is an eigenvector of $A$ corresponding to the eigenvalue $\lambda$, and $k$ is a non-zero constant, then $kX$ is also an eigenvector of $A$ corresponding to the eigenvalue $\lambda$.

**Example for Property 1**

Consider matrix $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ with eigenvalue $\lambda = 2 $. The eigenvectors corresponding to $\lambda = 2$ are of the form $X = \begin{bmatrix} x \\ 0 \end{bmatrix}$.

Let $X_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$ and $X_2 = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$. Both are eigenvectors of $A$ for $\lambda = 2$.

Their sum $X_1 + X_2 = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$ is also an eigenvector of $A$ for $\lambda = 2$.

**Example for Property 2**

Using the same matrix $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ with eigenvalue $\lambda = 2$, consider the eigenvector $X = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$.

If $k = 3$, then $kX = 3 \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}$ is also an eigenvector of $A$ for $\lambda = 2$.

#### How to Calculate Eigenvalues and Eigenvectors of a Square Matrix?


#### Example Matrix

Consider the matrix $A$:

$$ 
A = \begin{bmatrix} 
4 & 1 & 2 \\ 
0 & 3 & -1 \\ 
0 & 0 & 2 
\end{bmatrix} 
$$

#### Step 1: Calculate Eigenvalues

1. **Characteristic Equation**: Compute $\det(A - \lambda I) = 0$, where $I$ is the identity matrix and $\lambda$ is the eigenvalue.

2. **Compute the Determinant**:

   $$
   \det(A - \lambda I) = \det \begin{bmatrix} 
   4-\lambda & 1 & 2 \\ 
   0 & 3-\lambda & -1 \\ 
   0 & 0 & 2-\lambda 
   \end{bmatrix} 
   $$

   Since this is an upper triangular matrix, the determinant is the product of the diagonal elements:

   $$
   (4-\lambda)(3-\lambda)(2-\lambda) = 0
   $$

3. **Solve for Eigenvalues**: Solve the equation to find $\lambda_1 = 4$, $\lambda_2 = 3$, $\lambda_3 = 2$.

#### Step 2: Calculate Eigenvectors

For each eigenvalue, solve $(A - \lambda I)X = 0$:

1. **Eigenvalue $\lambda_1 = 4$**:

   $$
   (A - 4I) = \begin{bmatrix} 
   0 & 1 & 2 \\ 
   0 & -1 & -1 \\ 
   0 & 0 & -2 
   \end{bmatrix} 
   $$

   Solve $(A - 4I)X = 0$ to find the eigenvector $X_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}$.

2. **Eigenvalue $\lambda_2 = 3$**:

   $$
   (A - 3I) = \begin{bmatrix} 
   1 & 1 & 2 \\ 
   0 & 0 & -1 \\ 
   0 & 0 & -1 
   \end{bmatrix} 
   $$

   Solve $(A - 3I)X = 0$ to find the eigenvector $X_2 = \begin{bmatrix} -1 \\ 1 \\ 0 \end{bmatrix}$.

3. **Eigenvalue $\lambda_3 = 2$**:

   $$
   (A - 2I) = \begin{bmatrix} 
   2 & 1 & 2 \\ 
   0 & 1 & -1 \\ 
   0 & 0 & 0 
   \end{bmatrix} 
   $$

   Solve $(A - 2I)X = 0$ to find the eigenvector $X_3 = \begin{bmatrix} -3 \\ 2 \\ 2 \end{bmatrix}$.


```python
import numpy as np

# Define the matrix A
A = np.array([
    [4, 1, 2],
    [0, 3, -1],
    [0, 0, 2]
])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Display the results
print("Eigenvalues:\n", eigenvalues)

print("\nEigenvectors:\n", eigenvectors)
```

    Eigenvalues:
     [4. 3. 2.]
    
    Eigenvectors:
     [[ 1.         -0.70710678 -0.72760688]
     [ 0.          0.70710678  0.48507125]
     [ 0.          0.          0.48507125]]
    

#### Reference
1. https://byjus.com/maths/eigen-values/
2. https://zhuanlan.zhihu.com/p/165382601

