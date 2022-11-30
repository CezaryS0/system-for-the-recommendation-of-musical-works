from Managers.DirectoryManager import DirectoryManager
from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.callbacks import ModelCheckpoint

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
        return train_x

    def autoencode(self,shape):
        
        input_img = Input(shape=shape)
        encoded = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(input_img)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        
        encoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2,2),padding='same')(encoded)
        encoded = Conv2D(filters=8,kernel_size=(3,3), activation='relu',padding='same')(encoded)
        encoded = MaxPooling2D((2, 2), padding='same')(encoded)
        encoder = Model(input_img,encoded)

        decoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(encoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=8,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=16,kernel_size=(3,3),activation='relu',padding='same')(decoded)
        decoded = UpSampling2D((2,2))(decoded)
        decoded = Conv2D(filters=1, kernel_size=(3,3), activation='sigmoid', padding='same')(decoded)
        decoder = Model(input_img,decoded)
        
        return decoder

    def build_autoencoder(self,dataset_path,model_save_path):
        self.dm.create_main_dir(model_save_path)
        spectrograms_array = self.numpy.read_sliced_spectrograms_file(dataset_path)
        train_x = self.prepare_data_for_training(spectrograms_array)
        shape = np.shape(train_x)
        shape = (shape[1],shape[2],1)
        epochs = 50
        autoencoder = self.autoencode(shape)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.summary()
        checkpoint = ModelCheckpoint(model_save_path+'/weight.h5', monitor='val_loss',save_best_only=True)
        autoencoder.fit(train_x,train_x,epochs=epochs, callbacks=[checkpoint])
        autoencoder.save(model_save_path+'/autoencoder.h5')
        return autoencoder