import tensorflow as tf
from tensorflow.keras import layers, models, datasets
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical

# Load and preprocess MNIST dataset
def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    #x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
    #x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train, 10)  # One-hot encode labels
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Define LeNet-5 architecture
def build_lenet5():
    return models.Sequential([
        layers.Conv2D(6, kernel_size=5, activation='tanh', input_shape=(28, 28, 1), padding='same'),
        layers.AvgPool2D((2,2)),
        layers.Conv2D(16, kernel_size=5, activation='tanh'),
        layers.AvgPool2D((2,2)),
        layers.Flatten(),
        layers.Dense(120, activation='tanh'),
        layers.Dense(84, activation='tanh'),
        layers.Dense(10, activation='softmax')
    ])

# Train and evaluate the model
(x_train, y_train), (x_test, y_test) = load_mnist()
model = build_lenet5()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_split=0.1, batch_size=64, epochs=10, verbose=1)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}")

# Classification report
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

# Plot training history
plt.figure(figsize=(12,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()
