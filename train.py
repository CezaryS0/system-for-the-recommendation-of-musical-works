from CNN.CNN_V1 import Encoder

enc = Encoder()
enc.train_model('Train_Data','slices/tempo_bpm.npy')