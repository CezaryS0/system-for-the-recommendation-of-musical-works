from CNN.CNN_V1 import Encoder

enc = Encoder()
enc.train_model('/content/drive/MyDrive/Train_Data','slices/tempo_bpm.npy')