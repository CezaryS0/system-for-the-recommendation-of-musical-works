from Model.Managers.DirectoryManager import DirectoryManager
from Model.Numpy.NumpyArray import NumpyArray
from Model.Managers.ModelManager import ModelManager
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt

class Autoencoder:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.numpy = NumpyArray()
        self.model = ModelManager()

    def prepare_data_for_training(self,spectograms_array):
        train_x = spectograms_array
        shape_train_x = np.shape(train_x)
        train_x = train_x.astype('float32') /255
        train_x = np.reshape(train_x,(shape_train_x[0],shape_train_x[1],shape_train_x[2],1))
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_x, test_size=0.05, shuffle=True)
        return train_x,test_x, train_y, test_y

    def autoencode(self,shape):
        
        input_img = Input(shape=shape)
        encoded = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(input_img)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        encoded = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        encoded = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        
        encoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        encoded = Conv2D(filters=8,kernel_size=(3,3), activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        encoded = Conv2D(filters=4,kernel_size=(3,3), activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)

        decoded = Conv2D(filters=4,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(decoded)
        decoder = Model(input_img,decoded)
        
        return decoder

    def build_autoencoder(self,dataset_path,model_save_path):
        self.dm.create_main_dir(model_save_path)
        spectrograms_array = self.numpy.read_sliced_spectrograms_file(dataset_path)
        train_x,test_x, train_y, test_y = self.prepare_data_for_training(spectrograms_array)
        shape = np.shape(train_x)
        shape = (shape[1],shape[2],1)
        epochs = 3
        autoencoder = self.autoencode(shape)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',metrics=["acc"])
        autoencoder.summary()
        checkpoint = ModelCheckpoint(model_save_path+'/weight.h5', monitor='val_loss',save_best_only=True)
        history = autoencoder.fit(train_x,train_x,epochs=epochs,validation_split=0.1,validation_data=(test_x, test_y))
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(model_save_path+'/autoencoder_val_loss.png')
        autoencoder.save(model_save_path+'/autoencoder.h5')
        return autoencoder