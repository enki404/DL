import numpy as np
import pandas as pd
import sklearn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing


plt.rcParams["figure.figsize"] = (10, 10)

#################
### Load Data ###
#################

path = r"C:\Users\enki0\Desktop\CSCI5922\Final_Exam"

df = pd.read_csv(str(path+'\Labeled_News_Data_from_API_all.csv'),encoding='unicode_escape')
## Convert the labels from list to df
Labels_df = pd.DataFrame(df,columns=['LABEL'])
print('Input Data Shape:\n',df.shape)

### df_raw contains corpus ###
df_raw = pd.read_csv(r'C:\Users\enki0\Desktop\CSCI5922\Final_Exam\NewHeadlines_all.csv')
df_raw.reset_index(drop=True,inplace=True)
print('Input Raw Data Shape:\n',df.shape)

########################
### string to number ###
########################

# politics: 1, football: 0, science: 2
le = preprocessing.LabelEncoder()
le.fit(df.LABEL)
df['LABEL'] = le.transform(df.LABEL)



NumCols=df.shape[1]
print('DTM NumCols\n', NumCols)

top_words=list(df.columns[1:NumCols+1])

#########################
### Encoding the data ###
#########################

def Encode(review):
    words = review.split()
    if len(words) > 60:
        words = words[:60]
    encoding = []
    for word in words:
        try:
            # index in top_words
            index = top_words.index(word)
        except:
            index = (NumCols - 1)
        encoding.append(index)
    while len(encoding) < 60:
        encoding.append(NumCols)
    return encoding

###################################
# encode all of our reviews - which are called "text" Using vocab 
# convert (text) into numerical form 
# Replacing each word with its corresponding integer index value from the 
# vocabulary. Words not in the vocab will
# be assigned  as the max length of the vocab + 1 
## #################################

# Encode our training and testing datasets with same vocab. 

# training_data = np.array([Encode(review) for review in TrainData["text"]])

##################################
### OneHot Encoded Label for y ###
##################################

# number of labels: how many types of label in raw dataset
################
### Actual y ###
################

### y should be column vector, so transpose is considered ###

y = np.array([df['LABEL']]).T
print("Actual label y shape:\n", y.shape)

# Encode our training and testing datasets with same vocab. 

encoded_data_vocab = np.array([Encode(review) for review in df_raw["Content"]])
print("Encoded text data via vocabulary\n", encoded_data_vocab.shape)


# actual_label_y = y

# temp = y
# # number of rows in dataset
# n = np.size(df, 0)
# # num of categrical class
# numoflabel = 3
# onehot_labels = np.zeros((n,numoflabel))

# for i in range(n):
#     onehot_labels[i,temp[i]-1] = 1

# y = onehot_labels
# print("One Hot Encoded y shape is:\n", y.shape)

X = np.array(encoded_data_vocab)
#X = preprocessing.normalize(X)
y = np.array([df.iloc[:,0]]).T
# unique, counts = np.unique(y_train, return_counts=True)

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.2, shuffle= True, random_state=123)
#x_train,x_val,y_train,y_val = train_test_split(x_train1,y_train1, test_size = 0.05, shuffle= True, random_state=10)

print("The shape of x_test is \n", x_test.shape)
print("The shape of x_train is \n", x_train.shape)
print("The shape of y_test is \n", y_test.shape)
print("The shape of y_train is \n", y_train.shape)

#################
### CNN Model ###
#################
## input_dim: Integer. Size of the vocabulary
## input_length: Length of input sequences, when it is constant.

input_dim = NumCols + 1


model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=8, input_length=60),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.65),
  #tf.keras.layers.Conv1D(4, kernel_size=3, activation='relu',padding="same"),
 # tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Conv1D(4, kernel_size=3, activation='relu',padding="same"),
  tf.keras.layers.MaxPool1D(pool_size=2),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dropout(0.6),
  
  tf.keras.layers.Flatten(),
 # tf.keras.layers.Dropout(0.3),
 
  tf.keras.layers.Dense(3, activation="softmax")
])
    
model.summary()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    ## False: encoded 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ["accuracy"],
    optimizer = opt
    ) 


Hist=model.fit(x_train, y_train, batch_size=32, epochs=1500, validation_data=(x_test,y_test),shuffle=True)


### History and Accuracy ###
plt.figure(figsize=(5,5))
plt.plot(Hist.history['accuracy'],label='accuracy')
plt.plot(Hist.history['val_accuracy'],label='val_accuracy')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Accuracy',fontsize=15)
plt.title('CNN Accuracy Graph',fontsize=15)
#plt.ylim([0.5,1])
plt.legend(loc='lower right', fontsize=10)

plt.figure(figsize=(5,5))
plt.plot(Hist.history['loss'],label='loss')
plt.plot(Hist.history['val_loss'],label='val_loss')
plt.xlabel('Epoch',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.title('CNN Loss Graph',fontsize=15)
#plt.ylim([0.5,1])
plt.legend(loc='upper right', fontsize=10)


##################
### Save Model ###
#model.save("P3_CNN_Model")

####################
### Predictioins ###

print("The label of this test data is \n", y_test[0])  

predictions = model.predict([x_test])
print('The shape of all predictions for test data', predictions.shape)
print("The single prediction vector for x_test[0] is \n", predictions[0]) 
print("The max - final prediction label for x_test[2] is\n", np.argmax(predictions[0])) 
## The argmax of the 1st prediction - this is the label

##########################################################
### Confusion Matrix and Accuracy - and Visual Options ###
##########################################################

test_loss, test_accuracy = model.evaluate(x_test,y_test)
print("The test accuracy is \n", test_accuracy)

import seaborn as sns
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

Max_Values = np.squeeze(np.array(predictions.argmax(axis=1)))
#print(Max_Values)
#print(np.argmax([predictions]))
#print(confusion_matrix(Max_Values, y_test))

class_names = ['0:football',
               '1:politics',
               '2:science']
#cm = confusion_matrix(y_test, Max_Values, labels=labels)
cm1 = confusion_matrix(y_test, Max_Values, labels=[0,1,2])
print('cm is\n', cm1)

fig, ax = plt.subplots(figsize=(15,15)) 
#ax= plt.subplot()
#sns.set(font_scale=3)
#sns.set (rc = {'figure.figsize':(40, 40)})
sns.heatmap(cm1, annot=True, fmt='g', ax=ax, annot_kws={'size': 36})
ax.set_xlabel('True labels',fontsize = 25) 
ax.set_ylabel('Predicted labels',fontsize = 25)
ax.set_title('Confusion Matrix: Part3_CNN',fontsize = 36) 
ax.xaxis.set_ticklabels(class_names,rotation=90, fontsize = 25)
ax.yaxis.set_ticklabels(class_names,rotation=0, fontsize = 25)

cm1 = confusion_matrix(y_test, Max_Values, labels=[0,1,2])
cm1_disp = ConfusionMatrixDisplay(
    confusion_matrix=cm1, display_labels=class_names)
cm1_disp.plot()
cm1_disp.ax_.set_title("Part3 CNN Confusion Matrix",fontsize=35)
cm1_disp.ax_.set_ylabel("Predicted labels", fontsize=35)
cm1_disp.ax_.set_xlabel("True labels", fontsize=35)
plt.show()


