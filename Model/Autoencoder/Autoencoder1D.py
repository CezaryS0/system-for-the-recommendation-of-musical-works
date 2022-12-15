from Managers.DirectoryManager import DirectoryManager
from Numpy.NumpyArray import NumpyArray
from Managers.ModelManager import ModelManager
import numpy as np
from keras.models import Model
from keras.layers import Input,InputLayer
from keras.layers import Dense,Conv1D,MaxPooling1D,UpSampling1D,Cropping1D
from keras.callbacks import ModelCheckpoint

class Autoencoder1D:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.numpy = NumpyArray()
        self.model = ModelManager()

        
    def prepare_data_for_training(self,fusion):
        train_x  = fusion
        train_x = train_x.astype('float32')/255
        shape = np.shape(train_x)
        train_x = train_x.reshape(shape[0],shape[1], 1)
        return train_x

    def autoencode(self,shape):
        
        input_img = Input(shape=shape)
        encoded = Conv1D(filters=64,kernel_size=3,activation='relu',padding='same')(input_img)
        encoded = MaxPooling1D(2,padding='same')(encoded)
        encoded = Conv1D(filters=32,kernel_size=3,activation='relu',padding='same')(encoded)
        encoded = MaxPooling1D(2,padding='same')(encoded)
        encoded = Conv1D(filters=16,kernel_size=3,activation='relu',padding='same')(encoded)
        encoded = MaxPooling1D(2,padding='same')(encoded)
        encoded = Conv1D(filters=8,kernel_size=3,activation='relu',padding='same')(encoded)
        encoded = MaxPooling1D(2,padding='same')(encoded)
        encoded = Conv1D(filters=8,kernel_size=3, activation='relu',padding='same')(encoded)
        encoded = MaxPooling1D(2, padding='same')(encoded)
        encoder = Model(input_img,encoded)
        
        decoded = Conv1D(filters=8,kernel_size=3, activation='relu',padding='same')(encoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=8,kernel_size=3, activation='relu',padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=16,kernel_size=3, activation='relu',padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=32,kernel_size=3, activation='relu',padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=64,kernel_size=3, activation='relu',padding='same')(decoded)
        decoded = UpSampling1D(2)(decoded)
        decoded = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(decoded)
        decoded = Cropping1D((0,np.shape(decoded)[1] - shape[0]))(decoded)
        decoder = Model(input_img,decoded)
        return decoder

    def build_secondary_autoencoder(self,dataset_path,model_save_path):
        self.dm.create_main_dir(model_save_path)
        fusion_array = self.numpy.read_numpy_file(dataset_path,'fusion.npy')
        train_x = self.prepare_data_for_training(fusion_array)
        shape = np.shape(train_x)
        shape = (shape[1],1)
        epochs = 50
        autoencoder = self.autoencode(shape)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        checkpoint = ModelCheckpoint(model_save_path+'/weight.h5', monitor='val_loss',save_best_only=True)
        autoencoder.fit(train_x,train_x,epochs=epochs, callbacks=[checkpoint],verbose=1)
        autoencoder.save(model_save_path+'/autoencoder_secondary.h5')
        return autoencoder
