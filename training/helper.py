from pprint import pprint
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_model():
    model = Sequential([
        layers.experimental.preprocessing.Rescaling(
            1./255, input_shape=(24, 24, 1)),
        layers.Conv2D(24, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(15)]
    )

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    return model
