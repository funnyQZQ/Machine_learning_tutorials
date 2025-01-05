### Support Vector Machine (SVM)

#### 1. Definition

Support Vector Machine (SVM) is a powerful supervised learning algorithm primarily used for classification problems. Its core idea is to find an optimal hyperplane in the feature space that maximizes the distance between samples of different classes and this hyperplane.

#### 2. Key Concepts

##### 2.1 Hyperplane
- An n-1 dimensional plane that separates data in an n-dimensional feature space.
- Can be represented as: $w^Tx + b = 0$
  - Where $w$ is the normal vector
  - $b$ is the bias term
  - $x$ is the input vector

##### 2.2 Margin
- The minimum distance from samples of different classes to the separating hyperplane.
- The goal of SVM is to maximize this margin.
- Mathematical expression: $\text{margin} = \frac{2}{||w||}$

##### 2.3 Support Vectors
- Training samples that are closest to the separating hyperplane.
- Determine the position and orientation of the final hyperplane.
- Have a significant impact on the model's prediction results.

##### 2.4 Kernel Function
- Used to map the original feature space to a higher-dimensional space.
- Common kernel functions include:
  - Linear kernel: $K(x_i,x_j) = x_i^T x_j$
  - Polynomial kernel: $K(x_i,x_j) = (x_i^T x_j + c)^d$
  - Gaussian kernel (RBF): $K(x_i,x_j) = \exp(-\gamma ||x_i-x_j||^2)$

##### 2.5 Soft Margin
- Allows some samples to be misclassified.
- Introduces slack variables $\xi_i$ to handle non-linearly separable cases.
- The optimization objective becomes: $\min \frac{1}{2}||w||^2 + C\sum_{i=1}^n \xi_i$
  - $C$ is the penalty parameter, controlling the model's tolerance.

#### 3. Mathematical Derivation

##### 3.1 Hard Margin SVM Derivation
For linearly separable cases, our goal is to find the classification hyperplane with the maximum margin.

1) For any point $x_i$ with label $y_i \in \{-1,+1\}$, we want:
   - When $y_i=+1$, $w^T x_i + b \geq +1$
   - When $y_i=-1$, $w^T x_i + b \leq -1$
   
   These conditions can be combined into one inequality:
   $y_i(w^T x_i + b) \geq 1$
   
   The "1" here indicates that we require the distance from the classification boundary to the hyperplane to be at least $\frac{1}{||w||}$. This distance can be derived as follows:
   - For any point $x$, the distance to the hyperplane $w^Tx + b = 0$ is: $\frac{|w^Tx + b|}{||w||}$
   - According to our constraint $y_i(w^T x_i + b) \geq 1$
   - For support vectors (points on the classification boundary), the equality holds, i.e., $|w^T x_i + b| = 1$
   - Therefore, the distance from support vectors to the hyperplane is $\frac{1}{||w||}$
   This ensures that the size of the classification margin is $\frac{2}{||w||}$ (the sum of distances from both sides of the boundary to the hyperplane).

2) Maximizing the margin is equivalent to minimizing $\frac{1}{2}||w||^2$, so the optimization problem can be written as:

   $$
   \min_{w,b} \frac{1}{2}||w||^2
   $$
   $$
   s.t. \quad y_i(w^T x_i + b) \geq 1, \quad i=1,2,...,n
   $$

3) Using the Lagrange Multiplier Method:  
   The Lagrange multiplier method is a way to solve constrained optimization problems. For the SVM problem:
   - Original optimization objective: $\min \frac{1}{2}||w||^2$
   - Constraint: $y_i(w^T x_i + b) \geq 1$
   
   Introduce Lagrange multipliers $\alpha_i \geq 0$, and construct the Lagrangian function:  
   $$L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^n \alpha_i[y_i(w^T x_i + b) - 1]$$
   
   Where:
   - The first term $\frac{1}{2}||w||^2$ is the original optimization objective
   - The second term is the product of the constraint and the Lagrange multipliers
   - The minus sign is because our constraint is in the form of greater than or equal to
   
According to Lagrange duality, the original problem can be transformed into a dual form. The meaning of duality is:
1. We can first minimize the original variables $w,b$, and then maximize the dual variables $\alpha$
2. Alternatively, we can first maximize $\alpha$, and then minimize $w,b$
3. These two solving orders are equivalent when the KKT conditions are satisfied

Therefore, the original problem can be written as:

$$
\min_{w,b} \max_{\alpha} L(w,b,\alpha) = \max_{\alpha} \min_{w,b} L(w,b,\alpha)
$$

The reason for maximizing $\alpha$:
1. The role of the Lagrange multipliers $\alpha$ is to penalize the violation of the constraint. When the constraint is not satisfied (i.e., $y_i(w^T x_i + b) < 1$), by increasing $\alpha$, the objective function can be made larger, forcing the optimization process to find a solution that satisfies the constraint
2. From a mathematical perspective, this is the inevitable result of transforming the original constrained optimization problem into an unconstrained optimization problem. Maximizing $\alpha$ in the dual problem ensures that the original problem's constraints are satisfied
3. If the constraint is satisfied, then $\alpha[y_i(w^T x_i + b) - 1] = 0$, and maximizing $\alpha$ will not affect the objective function; if the constraint is not satisfied, maximizing $\alpha$ will cause the objective function to tend towards infinity, which is not allowed in the optimization process

This transformation is based on the following principles:
- According to the theory of Lagrange duality, under the KKT conditions, the optimal values of the original problem and the dual problem are equal
- For each inequality constraint $y_i(w^T x_i + b) \geq 1$ in the original problem, we introduce non-negative Lagrange multipliers $\alpha_i \geq 0$
- By constructing the Lagrangian function $L(w,b,\alpha)$, we incorporate the constraints into the objective function
- Minimizing the original variables $(w,b)$ while maximizing the dual variables $\alpha$ ensures that:
  - If the original constraint is satisfied, then $\alpha_i[y_i(w^T x_i + b) - 1] = 0$
  - If the original constraint is not satisfied, maximizing will cause the objective function to tend towards infinity

By solving this problem, we can obtain the optimal solution to the original problem.

4) Solving the Dual Problem:
   We can derive the dual problem through the following steps:

   1. First, take the partial derivatives of the Lagrangian function with respect to $w$ and $b$ and set them to zero:
      
      $$
      \frac{\partial L}{\partial w} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0
      $$

      Resulting in: $w = \sum_{i=1}^n \alpha_i y_i x_i$
      
      $$ 
      \frac{\partial L}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0 
      $$

      Resulting in: $\sum_{i=1}^n \alpha_i y_i = 0$

   2. Substitute the expression for $w$ back into the Lagrangian function:  
      $$ 
      L(w,b,\alpha) = \frac{1}{2}||w||^2 - \sum_{i=1}^n \alpha_i[y_i(w^T x_i + b) - 1]
      $$
      
      $$
      = \frac{1}{2}(\sum_{i=1}^n \alpha_i y_i x_i)^T(\sum_{j=1}^n \alpha_j y_j x_j) - \sum_{i=1}^n \alpha_i[y_i((\sum_{j=1}^n \alpha_j y_j x_j)^T x_i + b) - 1]
      $$

   3. Simplify to obtain the dual problem:

   $$
   \max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n\sum_{j=1}^n \alpha_i\alpha_j y_i y_j x_i^T x_j
   $$
   $$
   s.t. \quad \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0
   $$