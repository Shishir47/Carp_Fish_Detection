import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Test Keras imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Keras components imported successfully.")
