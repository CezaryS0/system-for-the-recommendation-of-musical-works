#from CNN.CNN_V1 import Encoder
#from CNN.CNN_V2 import Encoder
from CNN.CNN_V3 import Encoder

enc = Encoder()
enc.train_model('Train_Data/Train','slices/duration.npy')