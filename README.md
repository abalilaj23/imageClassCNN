#  Learning an Image Classification Model from Scratch

This project demonstrates how to build, train, and evaluate a simple image classification model using the **Fashion MNIST** dataset and **TensorFlow/Keras**. The dataset consists of **70,000 grayscale images** of clothing items from **10 categories** such as shirts, trousers, and ankle boots.

---

##  Dataset

- **Source:** [Fashion MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)
- **Training Set:** 60,000 images  
- **Test Set:** 10,000 images  
- **Image Size:** 28×28 pixels  
- **Categories:**

0 - T-shirt/top 5 - Sandal
1 - Trouser 6 - Shirt
2 - Pullover 7 - Sneaker
3 - Dress 8 - Bag
4 - Coat 9 - Ankle boot

yaml
Copy
Edit

---

##  Tools and Libraries

- **TensorFlow/Keras** – Deep learning framework  
- **NumPy, Pandas** – Data manipulation  
- **Matplotlib** – Data visualization  
- **Python** – Primary programming language

---

##  Project Workflow

### 1. Data Loading & Exploration
- Loaded using `tf.keras.datasets.fashion_mnist.load_data()`
- Split into training and test sets
- Visualized examples and checked label balance

### 2. Preprocessing
- Scaled pixel values from `[0, 255]` to `[0, 1]`
- Flattened 2D images to 1D vectors (for dense layers)
- Encoded class labels for model interpretation

---

##  Model Architecture

A basic feedforward neural network (Multilayer Perceptron):

```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),        # Flatten image to 784-dim vector
    keras.layers.Dense(128, activation='relu'),        # Hidden layer with ReLU
    keras.layers.Dense(64, activation='relu'),         # Another hidden layer
    keras.layers.Dense(10, activation='softmax')       # Output layer for 10 classes
])

```

## Training

- **Loss Function:** `SparseCategoricalCrossentropy`
- **Optimizer:** `Adam`
- **Metrics:** `accuracy`
- **Epochs:** 10–20 (adjustable)
- **Batch Size:** 32 or 64

###  Training Command

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_split=0.1)

```

##  Evaluation

After training:

-  Evaluated on **10,000 test images**
-  Plotted **confusion matrix** and **classification report**
-  Visualized **sample predictions** to inspect model performance

###  Evaluation Code

```python
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.2f}")

```

##  Key Takeaways

- Simple **feedforward networks** can achieve **~88–90% accuracy** on the Fashion MNIST dataset.
- **Preprocessing** and **visualization** are crucial for understanding data patterns and model behavior.
- Model performance can be further enhanced by incorporating:
  -  **Convolutional Neural Networks (CNNs)**
  -  **Dropout**
  -  **Data augmentation**
  -  **Regularization techniques**
