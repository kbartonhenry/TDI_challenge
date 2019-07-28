import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from imutils import paths
import argparse
import random
import pickle
import cv2
import os
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, concatenate, Input


print('Loading images...')
imagePaths = sorted(list(paths.list_images('Downloads/Dataset 2')))
random.seed(42)
random.shuffle(imagePaths)

#preprocess image data and encode labels
print('Extracting labels...')
data = []
labels = []
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (96, 96))
	image = img_to_array(image)
	data.append(image)
	l = imagePath.split(os.path.sep)[2]
	labels.append(l)
data = np.array(data, dtype="float") / 255.0
num_labels = pd.get_dummies(labels)

#split test and train
print('Making test/train split...')
(trainX, testX, trainY, testY) = train_test_split(data, num_labels, test_size=0.2, random_state=0)


#build NN
print('Creating CNN...')
#params
epochs = 23
high = 96
wide = 96
spectral_bands = 3
batch_size = 30

#structure 
inputxx = Input(shape=(high,wide,spectral_bands))
xx = Conv2D(filters=5, kernel_size=(3,3), padding='valid', data_format='channels_last', activation='relu')(inputxx)
xx = MaxPooling2D(pool_size=(2,2))(xx)
xx = Conv2D(filters=5, kernel_size=(3,3), padding='valid', data_format='channels_last', activation='relu')(xx)
xx = Conv2D(filters=5, kernel_size=(3,3), padding='valid', data_format='channels_last', activation='relu')(xx)
xx = MaxPooling2D(pool_size=(2,2))(xx)
xx = Flatten()(xx)
xx = Dense(50,activation='relu')(xx)
xx = Dropout(0.5)(xx)
xx = Dense(3,activation='softmax')(xx)
model = Model(inputs=inputxx, outputs=xx)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

#generator: augment data
datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True)
datagen.fit(trainX)

print('Training CNN...')
fitted_model = model.fit_generator(datagen.flow(trainX, trainY, batch_size=batch_size),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // batch_size,
	epochs=epochs, verbose=1)

#save model
model.save('Downloads/Dataset 2/cnn1.h5')




#plot accuracy on test vs. train over epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), fitted_model.history['acc'], label='train')
plt.plot(np.arange(0, epochs), fitted_model.history['val_acc'], label='validation')
plt.title("Accuracy over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



