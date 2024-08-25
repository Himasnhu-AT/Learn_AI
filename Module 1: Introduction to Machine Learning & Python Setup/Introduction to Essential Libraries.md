# Introduction to Essential Libraries

This tutorial introduces essential Python libraries for data manipulation, analysis, and visualization. We'll cover NumPy for numerical computing, Pandas for data handling and analysis, and Matplotlib for creating visualizations. You'll gain hands-on experience with basic operations and functionalities of these libraries.

## NumPy

NumPy (Numerical Python) is a fundamental library for scientific computing in Python. It provides powerful tools for working with arrays, matrices, and numerical operations.

### Creating Arrays

Sample Python Code: 

```{language}
import numpy as np

# Create a 1D array
array_1d = np.array([1, 2, 3, 4, 5])
print(array_1d)

# Create a 2D array (matrix)
array_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(array_2d)
```

### Array Operations

Sample Python Code: 

```{language}
# Basic arithmetic operations
print(array_1d + 2)  # Add 2 to each element
print(array_1d * 2)  # Multiply each element by 2
print(array_1d - array_1d[1:])  # Subtract elements

# Dot product
print(np.dot(array_1d, array_1d))

# Transpose
print(array_2d.transpose())
```

### Indexing and Slicing

Sample Python Code: 

```{language}
# Accessing elements
print(array_1d[0])  # First element
print(array_2d[1, 2])  # Element at row 1, column 2

# Slicing
print(array_1d[1:4])  # Elements from index 1 to 3 (exclusive)
```

### Other Useful Functions

- `np.zeros(shape)`: Create an array filled with zeros
- `np.ones(shape)`: Create an array filled with ones
- `np.arange(start, stop, step)`: Create an array of evenly spaced values
- `np.random.rand(shape)`: Create an array of random values
- `np.sum(array)`: Sum of all elements
- `np.mean(array)`: Average of all elements
- `np.std(array)`: Standard deviation of elements

## Pandas

Pandas is a powerful library for data analysis and manipulation. It provides data structures like Series and DataFrames, making it easier to work with structured data.

### Series

A Series is a one-dimensional labeled array capable of holding any data type.

Sample Python Code: 

```{language}
import pandas as pd

# Create a Series from a list
data = [1, 2, 3, 4, 5]
series = pd.Series(data)
print(series)

# Create a Series with custom labels
labels = ['a', 'b', 'c', 'd', 'e']
series = pd.Series(data, index=labels)
print(series)
```

### DataFrames

A DataFrame is a two-dimensional labeled data structure with rows and columns. It's similar to a spreadsheet.

Sample Python Code: 

```{language}
# Create a DataFrame from a dictionary
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 28], 'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)
print(df)

# Accessing data
print(df['Name'])  # Access column 'Name'
print(df.loc[0])  # Access row with index 0
print(df.iloc[1])  # Access row at position 1
```

### Data Manipulation

Pandas provides various methods for data manipulation, including:

- **Filtering:** Select rows based on conditions.
- **Sorting:** Sort rows based on specific columns.
- **Aggregation:** Calculate summary statistics (e.g., mean, sum, count).
- **Joining:** Combine multiple DataFrames.

Sample Python Code: 

```{language}
# Filter rows where Age is greater than 28
filtered_df = df[df['Age'] > 28]
print(filtered_df)

# Sort by Name
sorted_df = df.sort_values(by='Name')
print(sorted_df)

# Calculate the average Age
average_age = df['Age'].mean()
print(average_age)
```

## Matplotlib

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

### Basic Plots

Sample Python Code: 

```{language}
import matplotlib.pyplot as plt

# Line plot
x = np.arange(0, 10, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Sine Wave')
plt.show()

# Scatter plot
x = np.random.rand(100)
y = np.random.rand(100)
plt.scatter(x, y)
plt.title('Scatter Plot')
plt.show()
```

### Bar Plots and Histograms

Sample Python Code: 

```{language}
# Bar plot
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 30, 40]
plt.bar(categories, values)
plt.title('Bar Plot')
plt.show()

# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=20)
plt.title('Histogram')
plt.show()
```

### Customization

Matplotlib offers extensive customization options:

- **Line styles:** Solid, dashed, dotted, etc.
- **Marker styles:** Circle, square, triangle, etc.
- **Colors:** Named colors, RGB values, hex codes.
- **Labels and titles:** Add text to plots.
- **Legends:** Identify different data series.

## Assignments

1. **NumPy:**
    - Create a NumPy array of the first 10 even numbers.
    - Multiply the array by 5.
    - Calculate the sum of the array.

2. **Pandas:**
    - Create a DataFrame with columns "Name", "Age", and "City" containing data for 5 people.
    - Filter the DataFrame to show only people whose Age is greater than 25.
    - Calculate the average Age of all people in the DataFrame.

3. **Matplotlib:**
    - Create a line plot of the function y = x^2 from x = -5 to x = 5.
    - Create a scatter plot of 100 random points.
    - Create a bar chart showing the frequency of different letters in a given string.
