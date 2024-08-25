# Saving and Loading TensorFlow Models

## Introduction

This tutorial focuses on the crucial task of saving and loading trained TensorFlow models. This ability is essential for:

* **Reusing Models:** Avoid retraining from scratch, saving time and resources.
* **Deployment:** Prepare models for use in production environments or other applications. 
* **Continuing Training:**  Resume training from a specific checkpoint, allowing for gradual improvement or experimentation.

## Why Save Models?

Saving trained models is like preserving the knowledge and skills your model has acquired through training. You can then:

* **Deploy:** Integrate your model into various applications without needing the original training dataset or code.
* **Share:** Collaborate with others by easily sharing your trained models.
* **Resume Training:**  Continue training from a previous point, potentially incorporating new data or refining the model's performance.

## Saving a TensorFlow Model

Here's how you can save a TensorFlow model:

Sample Python Code: 

```{language}
import tensorflow as tf

# Define your model (example: a simple linear regression model)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile your model (specify optimizer, loss function, and metrics)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

# Train your model on your dataset
model.fit(X_train, y_train, epochs=10)

# Save the entire model to a file
model.save('my_model.h5')
```

**Explanation:**

1. **Import TensorFlow:** `import tensorflow as tf`
2. **Define Model:** Create your TensorFlow model using `tf.keras.Sequential` or other desired architecture.
3. **Compile Model:** Choose an optimizer, loss function, and metrics for model training.
4. **Train Model:** Use `model.fit()` to train your model on your dataset.
5. **Save Model:** Use `model.save()` to save the entire model to a file (here, `my_model.h5`). The `.h5` extension is commonly used for TensorFlow models.

## Loading a Saved Model

To use a saved model later, load it like this:

Sample Python Code: 

```{language}
# Load the saved model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Now you can use loaded_model for inference, evaluation, or continued training
```

**Explanation:**

1. **Load Model:** `tf.keras.models.load_model()` is used to load the saved model from the specified file path.
2. **Use Loaded Model:** Now you can use the `loaded_model` for tasks like:
   * **Inference:**  Making predictions on new data.
   * **Evaluation:**  Measuring the model's performance on a test set.
   * **Continued Training:**  Continuing training from the saved checkpoint, possibly with additional data.

## Saving and Loading Weights

Sometimes, you might only need to save the model's weights (parameters) and not the entire model architecture. This is useful when you already have the model architecture defined or when you want to swap out different model architectures while keeping the same weights.

Sample Python Code: 

```{language}
# ... (Define and train your model as before)

# Save just the weights
model.save_weights('my_model_weights.h5')
```

To load the weights:

Sample Python Code: 

```{language}
# ... (Define your model architecture)

# Load the weights into your model
model.load_weights('my_model_weights.h5')
```

**Explanation:**

* **Save Weights:** `model.save_weights()` saves only the weights to a file.
* **Load Weights:** `model.load_weights()` loads the saved weights into the specified model.

## Assignments

1. **Save and Load:**
   * Train a simple image classification model (e.g., using the MNIST dataset).
   * Save the trained model using `model.save()`.
   * Load the saved model using `tf.keras.models.load_model()`.
   * Evaluate the loaded model on the test set to ensure it works correctly.

2. **Save Weights:**
   * Train a more complex model (e.g., a convolutional neural network).
   * Save only the model's weights using `model.save_weights()`.
   * Create a new instance of the same model architecture.
   * Load the saved weights into the new model instance.
   * Evaluate the new model to confirm that the weights have been loaded correctly. 
