import cv2
import tensorflow as tf
import numpy as np
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from keras.models import Sequential
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def MultiClassClassificationModel():
    classifier = Sequential()
    classifier.add(Conv2D(64,kernel_size=(3,3),activation='relu',strides=(1,1)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(100,activation='relu'))
    classifier.add(Dense(50,activation='relu'))
    classifier.add(Dense(5,activation='softmax'))
    classifier.compile(loss = 'sparse_categorical_crossentropy',metrics = ['sparse_categorical_accuracy'],optimizer = 'adam')
    return classifier

afiles = r"Rice_Image_Dataset/Arborio/*"
bfiles = r"Rice_Image_Dataset/Basmati/*"
lfiles = r"Rice_Image_Dataset/lpsala/*"
jfiles = r"Rice_Image_Dataset/Jasmine/*"
kfiles = r"Rice_Image_Dataset/Karacadag/*"

dataset = []
labels = []

for files in glob.glob(afiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(0)

for files in glob.glob(bfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(1)

for files in glob.glob(lfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(2)

for files in glob.glob(jfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(3)

for files in glob.glob(kfiles):
    img = cv2.imread(files)
    dataset.append(img)
    labels.append(4)

dataset = np.array(dataset)
labels = np.array(labels)

X_train,X_test,Y_train,Y_test = train_test_split(dataset,labels,random_state=42,test_size=0.2)

model = MultiClassClassificationModel()

model.fit(X_train,Y_train,epochs = 10,validation_data = (X_test,Y_test))

model.save("ricepredictor.keras")

