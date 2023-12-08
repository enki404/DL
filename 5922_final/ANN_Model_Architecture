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

### ANN Model ###

ANN_model = tf.keras.models.Sequential([
    # input layer and 1st hidden layer
    tf.keras.layers.Dense(4, activation='sigmoid', input_shape = (4,)),
    # 2nd hidden layer
    tf.keras.layers.Dense(3, activation='relu'),
    # output layer
    tf.keras.layers.Dense(3, activation='softmax') 
    
    ])

print("Model summary is", ANN_model.summary())

### Compile Model ###

ANN_model.compile(
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
    optimizer = 'adam'
    ) 

from tensorflow.keras.utils import plot_model
plot_model(ANN_model,
           to_file=r'C:\Users\enki0\Desktop\CSCI5922\Final_Exam\ANN_model.png',
           show_shapes=True,
           show_layer_names=True
          )

