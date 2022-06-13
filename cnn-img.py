
"""""
CNN
"""""
##############  import pack #############
from tensorflow import keras
import numpy as np
import pylab as plt
import tensorflow as tf
from tensorflow.keras import layers ,models
from tensorflow.keras.layers import BatchNormalization
import os
import re
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import tensorflow.keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow.keras.metrics as km
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
import re
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from skimage.transform import rescale
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

########### Label Function #############
def read_label(filename):
    x = re.findall("a.._", filename)
    # print(x)
    action_id = x[0][1:3]
    return action_id

############## Load Data ################
label = []
d = os.listdir('data/ds_movies/')
f = []
data = []
len_f = []

############# images ############
for file in d:
 #   frames = []
   path = 'data/ds_movies/' + file
   label.append(int(read_label(file)))
   image=cv2.imread(path)
   image = rescale(image, 1, anti_aliasing=False)
   # image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
   image=cv2.resize(image,(240,180))
   data.append(image)


########## to array ##########
data_new=np.asarray(data)
print(data_new.shape)
# data_new=data_new.reshape(data_new.shape[0],data_new.shape[1],60,80,1)

########## label ###########
label_new1 = np.asarray(label)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = label_new1.reshape(len(label_new1), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
label_new = onehot_encoded
# print(label_new)

########## split data ##########
X_train, X_test, y_train, y_test = train_test_split(data_new, label_new,test_size=0.20, shuffle=True, random_state=0)
# print(X_train.shape)

########## function data augmentation ###########
def create_datagen():
    return ImageDataGenerator(
        # randamly expand or shrink between 1-zoom_range~1+zoom_range
        zoom_range=0.15,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        # fill in the margin with the color designated by cval
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        brightness_range=[0.3,1.0] #add by Daisuke
    )

############ Using original generator #############
BATCH_SIZE = 32
data_generator = create_datagen().flow(X_train, y_train, batch_size=BATCH_SIZE, seed=4019)
validation_generator = create_datagen().flow(X_test, y_test, batch_size=BATCH_SIZE, seed=2019)


# ################## Building a model #########################
ConvNN_model = models.Sequential()

########### conv 2D ################
# ConvNN_model.add(BatchNormalization(input_shape=(480,640,3)))
ConvNN_model.add(layers.Conv2D(8, (3, 3), activation='relu',padding='same'  ,input_shape=(180, 240, 3)))#
ConvNN_model.add(layers.Conv2D(8, (3, 3), activation='relu',padding='same'))
ConvNN_model.add(layers.Conv2D(8, (3, 3), activation='relu',padding='same'))
ConvNN_model.add(layers.MaxPooling2D((2, 2)))
ConvNN_model.add(Dropout(0.25))
ConvNN_model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='same'))
ConvNN_model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='same'))
ConvNN_model.add(layers.Conv2D(16, (3, 3), activation='relu',padding='same'))
ConvNN_model.add(layers.MaxPooling2D((2, 2)))
ConvNN_model.add(Dropout(0.2))
#
# ConvNN_model.add(layers.Conv2D(32, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.Conv2D(32, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.Conv2D(32, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.MaxPooling2D((2, 2)))
# ConvNN_model.add(Dropout(0.25))
# ConvNN_model.add(layers.Conv2D(256, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.Conv2D(256, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.Conv2D(256, (3, 3), activation='softmax',padding='same'))
# ConvNN_model.add(layers.MaxPooling2D((2, 2)))
# encode rows of matrix
# ConvNN_model.add(TimeDistributed(LSTM(64,activation='relu')))
# ConvNN_model.add(Dropout(0.2))
# # encode columns
# ConvNN_model.add(LSTM(64, activation='relu'))
ConvNN_model.add(layers.Flatten())
ConvNN_model.add(layers.Dense(100, activation='relu'))
ConvNN_model.add(layers.Dropout(0.25))
ConvNN_model.add(layers.Dense(80, activation='relu'))

ConvNN_model.add(layers.Dense(60,activation='relu'))
ConvNN_model.add(layers.Dense(40,activation='relu'))
ConvNN_model.add(layers.Dense(18,activation='softmax'))
ConvNN_model.summary()

########### conv 3D #############
# ConvNN_model.add(Conv3D(8, (1,3, 3), activation='relu',padding='same',input_shape=(299,60,80,3)))
# ConvNN_model.add(Conv3D(8, (1,3, 3), activation='relu',padding='same'))
# ConvNN_model.add(MaxPooling3D((1,2, 2)))
# ConvNN_model.add(Dropout(0.25))
# ConvNN_model.add(Conv3D(16, (1,3, 3), activation='relu',padding='same'))
# ConvNN_model.add(Conv3D(16, (1,3, 3), activation='relu',padding='same'))
# ConvNN_model.add(MaxPooling3D((1,2, 2)))
# ConvNN_model.add(Dropout(0.2))
# ConvNN_model.add(Flatten())
# ConvNN_model.add(layers.Dense(100, activation='relu'))
# ConvNN_model.add(layers.Dropout(0.25))
# # ConvNN_model.add(layers.Dense(80, activation='relu'))
# ConvNN_model.add(layers.Dense(60,activation='relu'))
# ConvNN_model.add(layers.Dense(40,activation='relu'))
# ConvNN_model.add(layers.Dense(18,activation='softmax'))
# ConvNN_model.summary()


################## Compiling a model ########################
import functools
top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=5)
top3_acc.__name__ = 'top3_acc'
top5_acc.__name__ = 'top5_acc'
optimiz=optimizers.Adam(learning_rate=0.001)
ConvNN_model.compile( optimiz,
        loss='categorical_crossentropy',
    # loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])#'top_k_categorical_accuracy'
                 # top3_acc ,top5_acc])



# ################## Fitting a model ##########################
# history=ConvNN_model.fit(x = X_train,
#           y = y_train,
#           epochs = 300,
#          batch_size=1 ,
#           validation_data = (X_test, y_test))

#

history = ConvNN_model.fit_generator(
    data_generator,
    steps_per_epoch=X_train.shape[0] / 32,
    epochs=300,
 validation_data=validation_generator)

########### evaluate #############
# result = ConvNN_model.evaluate(X_test, y_test)

result = ConvNN_model.evaluate_generator(validation_generator)
print("accuracy:",result[1],"loss:",result[0])
######### predict new labels ##############
labls_test = ConvNN_model.predict(X_test)
#labls_test = y_test.tolist()
# print("Predict labels:",labls_test,"labels test:",y_test)
# ConvNN_model.predict(X_test)
#print()

########## plot #################
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


