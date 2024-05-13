import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import numpy as np

# Load Olivetti dataset
data = fetch_olivetti_faces()
X_train, X_test, y_train, y_test = train_test_split(data.images, data.target, test_size=0.2, random_state=42)

# Preprocess the data
X_train = np.stack([X_train, X_train, X_train], axis=-1)  # Convert grayscale to RGB
X_test = np.stack([X_test, X_test, X_test], axis=-1)  # Convert grayscale to RGB
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Build the model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(40, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy}")
