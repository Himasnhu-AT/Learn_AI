# Logistic Regression

## Introduction

This tutorial dives into the world of logistic regression, a powerful classification algorithm used to predict categorical outcomes based on input features. We'll cover the core concepts, implementation, and practical applications of this widely-used machine learning technique.

## The Logistic Function

At the heart of logistic regression lies the **logistic function**, also known as the sigmoid function. This function takes any real-valued input and squashes it to a value between 0 and 1. 

**Mathematical Form:**

```
σ(z) = 1 / (1 + exp(-z))
```

**Visualization:**

![Logistic Function Plot](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/256px-Logistic-curve.svg.png)

**Interpretation:**

The logistic function transforms a linear combination of input features (z) into a probability.  Values close to 1 indicate a high probability of belonging to one class, while values close to 0 suggest a high probability of belonging to the other class.

## Maximum Likelihood Estimation

To build a logistic regression model, we need to find the optimal parameters (coefficients) that best fit the data. This is achieved through **maximum likelihood estimation (MLE)**.

**The Goal:**

MLE aims to find the set of parameters that maximizes the likelihood of observing the given data. In simpler terms, we want to find the parameters that make the observed outcomes most probable.

**How it Works:**

1. **Define the likelihood function:** This function calculates the probability of observing the actual labels given the predicted probabilities from the model.
2. **Maximize the likelihood:** We use optimization algorithms like gradient descent to find the parameters that maximize the likelihood function.

## Interpreting Probabilities and Predictions

Once the model is trained, we can use it to make predictions on new data points. The output of a logistic regression model is a probability between 0 and 1.

**Decision Boundary:**

To make a binary classification, we typically set a threshold (usually 0.5).  Predictions above the threshold are classified as one class, and those below are classified as the other.

**Example:**

Let's say we're building a model to predict whether a customer will click on an ad. The model outputs a probability of 0.75. Using a threshold of 0.5, we would predict that the customer will click on the ad.

## Regularization

Overfitting is a common problem in machine learning where the model learns the training data too well and performs poorly on unseen data. Regularization techniques help prevent overfitting by adding a penalty term to the loss function.

**Common Regularization Techniques:**

* **L1 Regularization (Lasso):** Encourages sparsity by pushing some coefficients towards zero.
* **L2 Regularization (Ridge):** Shrinks the magnitude of coefficients, reducing the impact of outliers.

## Logistic Regression with Scikit-learn

Scikit-learn provides a convenient and powerful way to implement logistic regression.

**Example Code:**

Sample Python Code: 

```{language}
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Multi-class Logistic Regression

Logistic regression can be extended to handle multi-class classification problems using the **one-vs-rest** or **softmax** approaches.

**One-vs-Rest:**

This method trains multiple binary classifiers, one for each class. Each classifier distinguishes a specific class from the rest.

**Softmax:**

Softmax regression directly predicts probabilities for each class. It uses a generalized logistic function that outputs probabilities that sum to 1 across all classes.

## Assignments

**Assignment 1:**

Implement a logistic regression model in Python using Scikit-learn to predict whether a customer will purchase a product based on their age and income. 

**Assignment 2:**

Explore the effect of different regularization techniques (L1 and L2) on the performance of a logistic regression model using a dataset of your choice. Analyze the coefficients and compare the accuracy on training and testing sets.

**Assignment 3:**

Implement a multi-class logistic regression model to classify handwritten digits using the MNIST dataset. Compare the performance of one-vs-rest and softmax methods.
