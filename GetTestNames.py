from Numpy.NumpyArray import NumpyArray
import numpy as np
numpy = NumpyArray()

numpy_array = numpy.read_numpy_file('Train_Data/Test','title.npy')

title_arr = numpy_array.tolist()



for title in set(title_arr):
   print(str(title))

if 'Trough_11' in title_arr:
    print('Yes!')
else:
    print('No!')