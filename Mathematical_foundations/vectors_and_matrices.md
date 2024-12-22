### Vectors and Matrices

#### 1. Vectors


**Definition**: A vector is a one-dimensional array of numbers, which can represent a point in space.



```python
import numpy as np

vector1 = np.array([1.5, 2.3, 3.7, 4.2])
vector2 = np.array([0.8, 1.4, 2.9, 5.6])
```

**Operations**:


1. **Addition**: Adding corresponding elements of two vectors.  
$\mathbf{a} = [a_1, a_2, ..., a_n]$ and $\mathbf{b} = [b_1, b_2, ..., b_n]$  
$\mathbf{a} + \mathbf{b} = [a_1 + b_1, a_2 + b_2, ..., a_n + b_n]$


```python
add_result = vector1 + vector2
add_result
```




    array([2.3, 3.7, 6.6, 9.8])



2. **Scalar Multiplication**: Multiplying each element of a vector by a scalar.  
For a scalar $c$ and vector $\mathbf{a} = [a_1, a_2, ..., a_n]$:  
$c \times \mathbf{a} = [c \times a_1, c \times a_2, ..., c \times a_n]$


```python
scalar = 2
scalar_multiplication_result = scalar * vector1
scalar_multiplication_result
```




    array([3. , 4.6, 7.4, 8.4])



3. **Dot Product**:  A scalar representing the product of two vectors.   
$\mathbf{a} = [a_1, a_2, ..., a_n]$ and $\mathbf{b} = [b_1, b_2, ..., b_n]$  
$\mathbf{a} \cdot \mathbf{b} = a_1b_1 + a_2b_2 + ... + a_nb_n$  
$\mathbf{a} \cdot \mathbf{b} = |\mathbf{a}||\mathbf{b}|\cos \theta$  
Similarity Measurement, Projection Calculation, Angle Calculation  




```python
dot_product_result = np.dot(vector1, vector2)
dot_product_result
```




    38.67



4. **Cross Product**: A vector perpendicular to two given vectors (in 3D space).  
$\mathbf{a} = [a_1, a_2, a_3]$ and $\mathbf{b} = [b_1, b_2, b_3]$  
$\mathbf{a} \times \mathbf{b} = [a_2b_3 - a_3b_2, a_3b_1 - a_1b_3, a_1b_2 - a_2b_1]$   
$\mathbf{a} = [a_1, a_2]$ and $\mathbf{b} = [b_1, b_2]$   
$\mathbf{a} \times \mathbf{b} = a_1b_2 - a_2b_1$  
In three-dimensional geometry, the cross product of vector a and vector b results in a vector, more commonly known as the normal vector, which is perpendicular to the plane formed by vectors a and b.  
In two-dimensional space, the cross product has another geometric meaning: a × b equals the area of the parallelogram formed by vectors a and b.


```python
vector1 = np.array([1.5, 2.3, 3.7])
vector2 = np.array([0.8, 1.4, 2.9])
cross_product_result = np.cross(vector1, vector2)
cross_product_result
```




    array([ 1.49, -1.39,  0.26])




```python
vector1 = np.array([1.5, 2.3])
vector2 = np.array([0.8, 1.4])
cross_product_result = np.cross(vector1, vector2)
cross_product_result
```




    array(0.26)



#### 2. Matrices


**Definition**: A matrix is a two-dimensional array of numbers, used to represent linear transformations and systems of linear equations.


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

    matrix1: 
     [[1.2 2.3 3.4]
     [4.5 5.6 6.7]
     [7.8 8.9 9. ]]
    matrix2: 
     [[ 2.1  3.2  4.3]
     [ 5.4  6.5  7.6]
     [ 8.7  9.8 10.9]]
    

**Operations**:


1. **Addition**: Adding corresponding elements of two matrices.  
For two matrices $A = [a_{ij}]$ and $B = [b_{ij}]$, their sum $C = A + B = [c_{ij}]$, where $c_{ij} = a_{ij} + b_{ij}$





```python
add_result = matrix1 + matrix2
add_result
```




    array([[ 3.3,  5.5,  7.7],
           [ 9.9, 12.1, 14.3],
           [16.5, 18.7, 19.9]])



2. **Scalar Multiplication**: Multiplying each element of a matrix by a scalar.  
For a scalar $c$ and matrix $A = [a_{ij}]$, their product $C = cA = [c \times a_{ij}]$, where $c_{ij} = c \times a_{ij}$






```python
scalar = 2
scalar_multiplication_result = scalar * matrix1
scalar_multiplication_result
```




    array([[ 2.4,  4.6,  6.8],
           [ 9. , 11.2, 13.4],
           [15.6, 17.8, 18. ]])



3. **Matrix Multiplication**: Combining two matrices to form a new matrix.  
For two matrices $A = [a_{ij}]$ and $B = [b_{ij}]$, their product $C = AB = [c_{ij}]$, where $c_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$


```python
matrix_multiplication_result = np.dot(matrix1, matrix2)
matrix_multiplication_result
```




    array([[ 44.52,  52.11,  59.7 ],
           [ 97.98, 116.46, 134.94],
           [142.74, 171.01, 199.28]])



4. **Transpose**: Flipping a matrix over its diagonal.  
For a matrix $A = [a_{ij}]$, its transpose $A^T = [a_{ji}]$


```python
transpose_result = matrix1.T
print('matrix1: \n', matrix1)
print('transpose_result: \n', transpose_result)
```

    matrix1: 
     [[1.2 2.3 3.4]
     [4.5 5.6 6.7]
     [7.8 8.9 9. ]]
    transpose_result: 
     [[1.2 4.5 7.8]
     [2.3 5.6 8.9]
     [3.4 6.7 9. ]]
    

#### Reference
1. https://www.statlect.com/matrix-algebra/vectors-and-matrices



