# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:34:15 2019

@author: Jagan Mohan
"""

# Since we are involved with Images here we don't have any Data Preprocessing step here unlike our previous ANN
# But we will have Image Preprocessing after building our CNN layers

# Part 1 - Building the CNN

#Importing the required libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initiating our CNN
classifier = Sequential()

# Step- 1 Adding a convolution layer
classifier.add(Convolution2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
#I have used input size of 64*64 as I am working on a CPU and less input size leads to faster results

#Step - 2 Adding the Max Pool Layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step - 3 Flattening
classifier.add(Flatten())

#Step - 4 Full Connection
classifier.add(Dense(units= 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

#Compiling our CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Part 2 - Image Preprocessing
#Let us use the flow from directory example from keras documentation
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(   training_set,
                            steps_per_epoch=8000,
                            epochs=25,
                            validation_data=test_set,
                            validation_steps=2000)
