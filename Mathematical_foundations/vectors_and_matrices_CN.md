### 向量和矩阵

#### 1. 向量

**定义**：向量是一个一维的数字数组，可以表示空间中的一个点。

```python
import numpy as np

vector1 = np.array([1.5, 2.3, 3.7, 4.2])
vector2 = np.array([0.8, 1.4, 2.9, 5.6])
```

**运算**：

1. **加法**：将两个向量的对应元素相加。  
$\mathbf{a} = [a_1, a_2, ..., a_n]$ 和 $\mathbf{b} = [b_1, b_2, ..., b_n]$  
$\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, ..., a_n + b_n]$

```python
add_result = vector1 + vector2
add_result
```

2. **标量乘法**：将向量的每个元素乘以一个标量。  
对于标量 $c$ 和向量 $\mathbf{a} = [a_1, a_2, ..., a_n]$：  
$c \times \mathbf{a} = [c \times a_1, c \times a_2, ..., c \times a_n]$

```python
scalar = 2
scalar_multiplication_result = scalar * vector1
scalar_multiplication_result
```

3. **点积**：两个向量的乘积结果是一个标量。  
$\mathbf{a} = [a_1, a_2, ..., a_n]$ 和 $\mathbf{b} = [b_1, b_2, ..., b_n]$  
$\mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + ... + a_nb_n$  
$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos \theta$  
用于相似性测量、投影计算、角度计算

```python
dot_product_result = np.dot(vector1, vector2)
dot_product_result
```

4. **叉积**：在三维空间中，两个向量的叉积是一个垂直于这两个向量的向量。  
$\mathbf{a} = [a_1, a_2, a_3]$ 和 $\mathbf{b} = [b_1, b_2, b_3]$  
$\mathbf{a} \times \mathbf{b} = [a_2b_3 - a_3b_2, a_3b_1 - a_1b_3, a_1b_2 - a_2b_1]$  
在二维空间中，叉积的几何意义是：a × b 等于由向量 a 和 b 形成的平行四边形的面积。

```python
vector1 = np.array([1.5, 2.3, 3.7])
vector2 = np.array([0.8, 1.4, 2.9])
cross_product_result = np.cross(vector1, vector2)
cross_product_result
```

```python
vector1 = np.array([1.5, 2.3])
vector2 = np.array([0.8, 1.4])
cross_product_result = np.cross(vector1, vector2)
cross_product_result
```

#### 2. 矩阵

**定义**：矩阵是一个二维的数字数组，用于表示线性变换和线性方程组。

```python
matrix1 = np.array([[1.2, 2.3, 3.4],
                    [4.5, 5.6, 6.7],
                    [7.8, 8.9, 9.0]])

matrix2 = np.array([[2.1, 3.2, 4.3],
                    [5.4, 6.5, 7.6], 
                    [8.7, 9.8, 10.9]])

print('matrix1: \n', matrix1)
print('matrix2: \n', matrix2)
```

**运算**：

1. **加法**：将两个矩阵的对应元素相加。  
对于两个矩阵 $A = [a_{ij}]$ 和 $B = [b_{ij}]$，它们的和 $C = A + B = [c_{ij}]$，其中 $c_{ij} = a_{ij} + b_{ij}$

```python
add_result = matrix1 + matrix2
add_result
```

2. **标量乘法**：将矩阵的每个元素乘以一个标量。  
对于标量 $c$ 和矩阵 $A = [a_{ij}]$，它们的乘积 $C = cA = [c \times a_{ij}]$，其中 $c_{ij} = c \times a_{ij}$

```python
scalar = 2
scalar_multiplication_result = scalar * matrix1
scalar_multiplication_result
```

3. **矩阵乘法**：将两个矩阵结合形成一个新矩阵。  
对于两个矩阵 $A = [a_{ij}]$ 和 $B = [b_{ij}]$，它们的乘积 $C = AB = [c_{ij}]$，其中 $c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$

```python
matrix_multiplication_result = np.dot(matrix1, matrix2)
matrix_multiplication_result
```

4. **转置**：将矩阵沿其对角线翻转。  
对于矩阵 $A = [a_{ij}]$，其转置 $A^T = [a_{ji}]$

```python
transpose_result = matrix1.T
print('matrix1: \n', matrix1)
print('transpose_result: \n', transpose_result)
```

#### 参考文献
1. https://www.statlect.com/matrix-algebra/vectors-and-matrices