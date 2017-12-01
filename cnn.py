# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the cnn
classifier=Sequential()
#adding layers
classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#second layer
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#flattening
classifier.add(Flatten())
#ann framework full connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
#compiling our model
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#image preprocessing
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=1,
        validation_data=test_set,
        validation_steps=2000)
#making a new prediction
import numpy as np
from keras.preprocessing import image
img = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
x=classifier.predict(img)
training_set.class_indices
if  x[0][0] == 1:
    prediction="dog"
else:
    prediction="cat"
print(prediction)