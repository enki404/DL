import tempfile
import os
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers.legacy import Adam
import zipfile

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()


# Normalize the input image so that each pixel value is between 0 and 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

input_shape = train_images.shape + (1,)

# Define the model architecture.
model = keras.Sequential([
  keras.layers.Conv2D(input_shape = input_shape[1::],filters=32, kernel_size=(3, 3), activation='relu'),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.L2(0.07))
])


model.summary()

# SparseCC not equal to CC, sparseCC can use output as (None,10), CC retruns as (None,1)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics = ['accuracy'])

Hist = model.fit(train_images, train_labels, batch_size=128,epochs =50, validation_split=0.1)

unpruned_test_accuracy = model.evaluate(test_images,test_labels)

print('Unpruned test accuracy:', unpruned_test_accuracy)

_, keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model,keras_file,include_optimizer=True)

_, zip1 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip1, 'w',compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
   # print("Size of unpruned model before compression: %.2f Mb" % (os.path.getsize(keras_file)/(2**20)))
        
    print("Size of unpruned model After compression: %.2f Mb" % (os.path.getsize(zip1) / float(2**20)))
        


# def setup_pretrained_weights():

#   model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       optimizer='adam',
#       metrics=['accuracy']
#   )

#   model.fit(train_images, train_labels)

#   _, pretrained_weights = tempfile.mkstemp('.tf')

#   model.save_weights(pretrained_weights)

#   return pretrained_weights

# pretrained_weights = setup_pretrained_weights()

# model.load_weights(pretrained_weights) # optional but recommended.


# Train the digit classification model


loaded_model = tf.keras.models.load_model(keras_file)

# Compute end step to finish pruning after 2 epochs.
batch_size = 256
es = 50
validation_split = 0.1 # 10% of training set will be used for validation set. 

#num_images = int(train_images.shape[0] * (1 - validation_split))
num_images = int(train_images.shape[0]) 
end_step = np.ceil(1.0*num_images / batch_size).astype(np.int32) * es

# Define model for pruning.
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.8,
                                                               final_sparsity=0.8,
                                                               begin_step=0,
                                                               end_step=end_step)
      }


model_for_pruning = keras.Sequential([
    keras.layers.Conv2D(input_shape = input_shape[1::],filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    tfmot.sparsity.keras.prune_low_magnitude(keras.layers.Dense(10,activation='softmax',kernel_regularizer=tf.keras.regularizers.L2(0.07)), **pruning_params)
    ])


#tfmot.sparsity.keras.prune_low_magnitude(loaded_model, **pruning_params)

model_for_pruning.summary()

adam = Adam()

# `prune_low_magnitude` requires a recompile.
model_for_pruning.compile(
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              optimizer=adam)

logdir = tempfile.mkdtemp()

callbacks = [
  tfmot.sparsity.keras.UpdatePruningStep(),
  tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]


Hist2 = model_for_pruning.fit(
  train_images,
  train_labels,
  batch_size=256,
  epochs=50,
  callbacks = callbacks,
  validation_split=0.1,
  use_multiprocessing = True
)

pruned_model_accuracy = model_for_pruning.evaluate(
    test_images, test_labels, verbose=0)

_, pruned_keras_file = tempfile.mkstemp('.h5')
tf.keras.models.save_model(model_for_pruning,pruned_keras_file,include_optimizer=True)


_, zip2 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip2, 'w',compression=zipfile.ZIP_DEFLATED) as f:
        f.write(pruned_keras_file)
#print('Size of pruned model before compression: %.2f Mb'(os.path.getsize(pruned_keras_file)/float(2**20)))
        
print('Size of pruned model After compression: %.2f Mb' % (os.path.getsize(zip2)/float(2**20)))

print('pruned test accuracy:', pruned_model_accuracy)

# keras_file = tempfile.mkstemp('.h5')
# tf.keras.models.save_model(model, keras_file, include_optimizer=False)
# print('Saved baseline model to:', keras_file)

## History and Accuracy ###
plt.figure(figsize=(5,5))
plt.plot(Hist.history['accuracy'],label='Unpruned Model Accuracy')
plt.plot(Hist.history['val_accuracy'],label='Unpruned Model val_accuracy')
plt.plot(Hist2.history['accuracy'],label='Pruned Model accuracy')
plt.plot(Hist2.history['val_accuracy'],label='Pruned Model val_accuracy')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title('Accuracy Graph',fontsize=15)
#plt.ylim([0.5,1])
plt.legend(loc='lower right', fontsize=10)

plt.figure(figsize=(5,5))
plt.plot(Hist.history['loss'],label='Unpruned Model Loss')
plt.plot(Hist.history['val_loss'],label='Unpruned Model val_loss')
plt.plot(Hist2.history['loss'],label='Pruned Model Loss')
plt.plot(Hist2.history['val_loss'],label='Pruned Model val_loss')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.title('Loss Graph',fontsize=15)
#plt.ylim([0.5,1])
plt.legend(loc='upper right', fontsize=10)

%tensorboard --logdir={logdir}


