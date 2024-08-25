# Introduction to TensorFlow: Tensors, Variables, and Operations

This tutorial introduces you to the fundamental building blocks of TensorFlow, a powerful open-source library for numerical computation and large-scale machine learning. We'll cover the following concepts:

* **Tensors:** The core data structure in TensorFlow, representing multidimensional arrays.
* **Variables:** Mutable tensors used to store and update model parameters during training.
* **Operations:** Mathematical computations performed on tensors, enabling complex calculations.

## Tensors: The Foundation of TensorFlow

At its core, TensorFlow operates on **tensors**, which are multidimensional arrays. Think of them as generalizations of matrices to any number of dimensions.  Each dimension is called an **axis** or **rank**.

**Creating Tensors:**

Sample Python Code: 

```{language}
import tensorflow as tf

# Scalar (rank 0):
scalar = tf.constant(5)

# Vector (rank 1):
vector = tf.constant([1, 2, 3])

# Matrix (rank 2):
matrix = tf.constant([[1, 2], [3, 4]])

# Tensor (rank 3):
tensor = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(f"Scalar: {scalar.numpy()}")
print(f"Vector: {vector.numpy()}")
print(f"Matrix: \n{matrix.numpy()}")
print(f"Tensor: \n{tensor.numpy()}")
```

**Understanding Tensor Properties:**

* **Data Type:**  The type of data stored in the tensor (e.g., `int32`, `float32`, `string`).
* **Shape:** A tuple representing the size of each dimension (e.g., `(2, 3)` for a matrix with 2 rows and 3 columns).
* **Rank:** The number of dimensions in the tensor.

**Tensor Operations:**

TensorFlow provides numerous operations for manipulating tensors:

Sample Python Code: 

```{language}
# Arithmetic operations:
addition = tf.add(vector, [4, 5, 6])

# Element-wise multiplication:
multiplication = tf.multiply(matrix, 2)

# Transpose:
transpose = tf.transpose(matrix)

# Reshape:
reshaped_tensor = tf.reshape(tensor, (4, 2))

# Indexing and Slicing:
sliced_vector = vector[1:3]
```

## Variables: Mutable Tensors for Learning

**Variables** are tensors whose values can be modified. They are essential for storing and updating model parameters during training.

**Creating Variables:**

Sample Python Code: 

```{language}
# Initialize a variable with a value:
weight = tf.Variable(0.0)

# Modify the variable's value:
weight.assign(1.0) 

# Increment the variable:
weight.assign_add(0.5)
```

**Key Considerations:**

* **Initialization:** You must initialize variables with initial values.
* **Mutability:** Variables can be updated repeatedly, unlike constants.
* **Training:** Variables are crucial for storing parameters that are adjusted during model training.

## Operations: Computing with Tensors

TensorFlow operations perform mathematical computations on tensors, enabling complex computations for tasks like machine learning.

**Common Operations:**

Sample Python Code: 

```{language}
# Addition: tf.add(tensor1, tensor2)
# Subtraction: tf.subtract(tensor1, tensor2)
# Multiplication: tf.multiply(tensor1, tensor2)
# Division: tf.divide(tensor1, tensor2)
# Matrix Multiplication: tf.matmul(matrix1, matrix2)
# Reduction (e.g., sum, mean): tf.reduce_sum(tensor), tf.reduce_mean(tensor)
# Activation Functions (e.g., ReLU): tf.nn.relu(tensor)
```

## Assignment

1. **Tensor Manipulation:** Create a 3x3 matrix using `tf.constant`. Then:
    * Calculate the sum of its elements.
    * Extract the diagonal elements.
    * Transpose the matrix.
    * Reshape the matrix into a vector.
2. **Variable Usage:**  Create a variable initialized to 0. Then, update its value using `assign_add` five times, incrementing it by 1 each time.
3. **Operations:**  Create two vectors of size 5. Calculate their dot product and element-wise multiplication.

This tutorial has introduced you to the fundamental components of TensorFlow: tensors, variables, and operations. These building blocks form the foundation for constructing and training powerful machine learning models. Remember to experiment and explore different operations and techniques as you delve deeper into TensorFlow!
