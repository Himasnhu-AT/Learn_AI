# Data Cleaning: Handling Missing Values

## Introduction

Missing data is a pervasive problem in real-world datasets. It can arise due to various reasons, such as data entry errors, equipment malfunctions, or simply incomplete records. Ignoring missing data can lead to biased analysis and inaccurate results. This tutorial delves into techniques for identifying and handling missing values, empowering you to clean your data and extract meaningful insights.

## Identifying Missing Values

Before addressing missing values, you need to identify them. In Python, the `pandas` library provides powerful tools for this purpose. Let's illustrate with an example:

Sample Python Code: 

```{language}
import pandas as pd

# Sample data with missing values
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, None, 30, 28, 22],
        'City': ['New York', 'London', 'Paris', None, 'Tokyo']}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Identify missing values using `isnull()`
print(df.isnull())

# Count missing values in each column
print(df.isnull().sum())
```

**Output:**

```
      Name   Age       City
0    Alice  25.0   New York
1      Bob   NaN     London
2  Charlie  30.0      Paris
3    David  28.0       None
4      Eve  22.0      Tokyo

      Name    Age   City
0  False  False  False
1  False   True  False
2  False  False  False
3  False  False   True
4  False  False  False

Name     0
Age      1
City     1
dtype: int64
```

The `isnull()` method identifies missing values, while `isnull().sum()` provides a count for each column.

## Handling Missing Values

### 1. Deletion

- **Listwise Deletion:** Simply remove rows containing any missing value. This can be useful for small datasets with a few missing values but can lead to significant data loss if the missing values are widespread.

Sample Python Code: 

```{language}
# Delete rows with missing values
df_dropped = df.dropna()
print(df_dropped)
```

- **Pairwise Deletion:**  Remove rows with missing values only for the specific calculations involving those variables. This is less drastic but may introduce biases if missingness is not random.

### 2. Imputation

- **Mean/Median Imputation:** Replace missing values with the mean or median of the respective column. This assumes the missing values are similar to the existing values but can distort the distribution if the data is skewed.

Sample Python Code: 

```{language}
# Impute missing age with the mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)
print(df)
```

- **Mode Imputation:**  Replace missing values with the most frequent value in the column. Suitable for categorical variables but less effective for continuous data.

- **K-Nearest Neighbors (KNN) Imputation:**  Find the K nearest neighbors to the missing value based on other features and impute with the average of the neighbors' values. More sophisticated and accounts for relationships between features.

Sample Python Code: 

```{language}
from sklearn.impute import KNNImputer

# Impute missing values using KNN
imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
print(df_imputed)
```

- **Regression Imputation:**  Build a regression model to predict the missing values based on other features. Provides a more accurate estimate than simple mean/median but requires a strong correlation between features.

Sample Python Code: 

```{language}
from sklearn.linear_model import LinearRegression

# Impute missing age using linear regression
model = LinearRegression()
model.fit(df[['Age']], df['City'])
df['City'].fillna(model.predict(df[['Age']]), inplace=True)
print(df)
```

### 3. Other Strategies

- **Flag Missing Values:** Instead of replacing, create a separate indicator variable that flags rows with missing values. This allows you to analyze the missingness pattern and avoid distorting the original data.

Sample Python Code: 

```{language}
# Create a flag for missing values in 'City'
df['City_Missing'] = df['City'].isnull()
print(df)
```

- **Model-Based Approaches:** Advanced techniques like multiple imputation or maximum likelihood estimation can provide more robust and statistically sound solutions for handling missing data, especially when dealing with complex relationships.

## Choosing the Right Approach

The best approach for handling missing values depends on:

- **Nature of Missing Data:** Is it Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR)?
- **Data Type:** Is it continuous, categorical, or ordinal?
- **Problem Context:** What are the goals of your analysis, and how sensitive are your results to missing data?

## Assignments

1. **Explore a real-world dataset with missing values.** Identify the missing values, analyze their pattern, and apply different imputation techniques.
2. **Compare the performance of different imputation methods** on a specific dataset and evaluate their impact on your analysis. 
3. **Implement a model-based approach** to handle missing data using a library like `mice` or `fancyimpute`.

By mastering these techniques, you equip yourself with the tools to clean and prepare your data for insightful analysis. 
