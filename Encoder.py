from NumpyArray import NumpyArray
import librosa
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D

class Encoder:

    def __init__(self):
        pass

    def encode(self,main_output_folder):
        numpy = NumpyArray()
        numpy_array = numpy.read_spectrograms_to_numpy_array(main_output_folder)
        spec = librosa.amplitude_to_db(numpy_array,ref=np.max)

        x_train = spec.astype('float32') /255.
        x_train = np.reshape(x_train,(1,128,1291,1))

        input_img = Input(shape=(128,1291,1))

        encoded = Conv2D(filters=16,kernel_size=(3,3), input_shape=(128,1291,1),activation='relu',padding='same')(input_img)
        encoded = MaxPooling2D()
        
ec = Encoder()
ec.encode("Train_Spectrogram_Images")