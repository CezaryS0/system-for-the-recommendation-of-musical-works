from Numpy.NumpyArray import NumpyArray

numpy = NumpyArray()

numpy_array = numpy.read_numpy_file('Train_Data/Test/slices','title.npy')

title_arr = numpy_array.tolist()

for title in set(title_arr):
    print(str(title))