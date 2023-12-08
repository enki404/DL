import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

### CNN Model ###

CNN_model = tf.keras.models.Sequential([
    #(None, 30,30,1)
    # filter: the dimension of the output space (the number of filters in the convolution).
    tf.keras.layers.Conv2D(input_shape=(30,30,1),filters=2,kernel_size=(3,3),strides=(1,1),activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=4,kernel_size=(3,3),strides=(1,1), activation="relu",padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
    
    ])

print("Model summary is",CNN_model.summary())

### Compile Model ###

CNN_model.compile(
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
    optimizer = 'adam'
    ) 

from tensorflow.keras.utils import plot_model
plot_model(CNN_model,
            to_file=r'C:\Users\enki0\Desktop\CSCI5922\Final_Exam\CNN_model.png',
            show_shapes=True,
            show_layer_names=True
          )

