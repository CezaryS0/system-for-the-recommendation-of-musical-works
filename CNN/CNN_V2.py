from statistics import mode
from Numpy.NumpyArray import *
import pandas as pd
import numpy as np
from keras import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import np_utils
from Managers.DirectoryManager import DirectoryManager
from sklearn.model_selection import train_test_split

class Encoder:

    def __init__(self):
        self.dm = DirectoryManager()
        self.numpy = NumpyArray()
        self.encoder = None

    def prepare_data_for_training(self,spectrograms_array,labels):
        train_x, test_x, train_y, test_y = train_test_split(spectrograms_array, labels, test_size=0.05, shuffle=True)
        train_y = np_utils.to_categorical(train_y)
        shape_train_x = np.shape(train_x)
        shape_test_x = np.shape(test_x)
        labels_shape = np.shape(train_y)
        train_x = np.reshape(train_x,(shape_train_x[0],shape_train_x[1],shape_train_x[2],1))
        test_x = np.reshape(test_x,(shape_test_x[0],shape_test_x[1],shape_test_x[2],1))
        train_x = train_x /255
        test_x = train_x /255
        test_y = np_utils.to_categorical(test_y, num_classes=labels_shape[1])
        return train_x, test_x, train_y, test_y

    def encode(self,shape,classes) -> Model:
        input_img = Input(shape=shape)
   
        encoded = Sequential()
        encoded = Conv2D(filters=64,kernel_size=(3,3),padding='same',activation='relu')(input_img)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=256,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,4))(encoded)
        encoded = Conv2D(filters=512,kernel_size=(3,3),padding='same',activation='relu')(encoded)
        #encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D((2,2))(encoded)
        encoded  = Flatten()(encoded)
        encoded = Dense(64)(encoded,activation="relu")
        encoded = Dense(32)(encoded,activation="relu")
        encoded = Dense(16)(encoded,activation="relu")
        encoded = Dense(1)(encoded)
        encoder = Model(inputs=input_img,outputs=encoded)
        print(encoder.output_shape)
        return encoder

    def train_model(self,dataset_path,feature_file):
        self.dm.create_main_dir("Saved_Model")
        spectrograms_array = self.numpy.read_sliced_spectrograms_file(dataset_path)
        labels = self.numpy.read_numpy_file(dataset_path,feature_file)
        labels = np.reshape(labels,(np.shape(labels)[0],1))
        train_x, test_x, train_y, test_y = self.prepare_data_for_training(spectrograms_array,labels)
        shape_train_x = np.shape(train_x)
        input_shape = (shape_train_x[1],shape_train_x[2],1)
        self.encoder = self.encode(input_shape,np.shape(train_y)[1])
        self.encoder.compile(loss="mse", optimizer="Adam", metrics=['accuracy'])
        pd.DataFrame(self.encoder.fit(train_x, train_y, epochs=10, validation_split=0.1).history).to_csv("Saved_Model/training_history.csv")
        self.encoder.evaluate(test_x,test_y)