import os 
import numpy as np
from Numpy.TrainingData import TrainingData
from Managers.JSONManager import JSONManager
from Managers.DirectoryManager import DirectoryManager
from Utilities.Utils import Utils

class NumpyArray:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.ut = Utils()

    def assingValuesToLists(self,data,json_file_data,matrix):
        
        for name in json_file_data:
            if name not in data.dataDict:
                data.dataDict[name]=list()
                data.fields.append(name)
            data.dataDict[name].append(json_file_data[name])
        data.spectrograms.append(matrix)

    def read_full_spectrograms_to_array(self,main_output_folder):
        data = TrainingData()
        json = JSONManager()
        for folder in os.scandir(main_output_folder):
            if folder.is_dir:
                dirPath = os.path.join(main_output_folder,folder.name)
                matrix = self.ut.read_image_to_numpy(os.path.join(dirPath,folder.name+".png"))
                jsonPath = os.path.join(dirPath,folder.name+".json")
                json.file_open(jsonPath,'r')
                json_file_data = json.read_JSON()
                self.assingValuesToLists(data,json_file_data,matrix)
                json.closeFile()

        return data

    def read_sliced_spectrograms(self,main_output_folder):
        data = TrainingData()
        json = JSONManager()
        for folder in os.scandir(main_output_folder):
            if folder.is_dir:
                jsonPath = os.path.join(main_output_folder,folder.name,folder.name+".json")
                dirPath = os.path.join(main_output_folder,folder.name,'slices')
                json.file_open(jsonPath,'r')
                json_file_data = json.read_JSON()
                for file in self.dm.get_all_files_in_dir(dirPath):
                    matrix = self.ut.read_image_to_numpy(os.path.join(dirPath,file))
                    self.assingValuesToLists(data,json_file_data,matrix)
                json.closeFile()
        return data
                
    def reshape_images(self,data):
        reshaped_list = []
        size_list = []
        for elem in data.spectrograms:
            x,y = np.shape(elem)
            size_list.append([x,y])
            reshaped_list.append(np.reshape(elem,x*y))
        return reshaped_list,size_list

    def save_array_to_numpy_file(self,arr,path):
        if type(arr[0]) is float:
            np.save(path,np.asarray(arr,dtype=np.float32))
        else:
            np.save(path,np.asarray(arr))

    def save_detail_to_numpy_files(self,data,path_train,path_test,train_size,test_size):
        buf_array = []
        for key in data.dataDict:
            save_path_train = os.path.join(path_train,key+'.npy')
            save_path_test = os.path.join(path_test,key+'.npy')
            buf_array.clear()
            for i in range(train_size):
                buf_array.append(data.dataDict[key][i])
            self.save_array_to_numpy_file(buf_array,save_path_train)
            buf_array.clear()
            for i in range(train_size,train_size+test_size):
                buf_array.append(data.dataDict[key][i])
            self.save_array_to_numpy_file(buf_array,save_path_test)

    def save_full_spectrograms(self,main_output_folder,data_full):
        save_path_train_spectrograms = os.path.join(main_output_folder,'Train',"spectrograms.npy")
        save_path_train_dims = os.path.join(main_output_folder,'Train',"spectrograms_dims.npy")
        save_path_test_spectrograms = os.path.join(main_output_folder,'Test',"spectrograms.npy")
        save_path_test_dims = os.path.join(main_output_folder,'Test',"spectrograms_dims.npy")
        spectrogram_array,size_array = self.reshape_images(data_full)
        train_size = int(len(spectrogram_array)*0.9)
        test_size = len(spectrogram_array)-train_size
        buf_array = []
        buf_size_array = []
        for i in range(train_size):
            buf_array.append(spectrogram_array[i])
            buf_size_array.append(size_array[i])
        np.save(save_path_train_spectrograms,np.array(buf_array,dtype=object))
        np.save(save_path_train_dims,np.array(buf_size_array,dtype=object))
        buf_array.clear()
        buf_size_array.clear()
        for i in range(train_size,train_size+test_size):
            buf_array.append(spectrogram_array[i])
            buf_size_array.append(size_array[i])
        np.save(save_path_test_spectrograms,np.array(buf_array,dtype=object))
        np.save(save_path_test_dims,np.array(buf_size_array,dtype=object))
        self.save_detail_to_numpy_files(data_full,
        os.path.join(main_output_folder,'Train'),
        os.path.join(main_output_folder,'Test'),
        train_size,test_size)
      

    def save_slice_spectrograms(self,main_output_folder,data_sliced):
        save_dir_train = os.path.join(main_output_folder,'Train','slices')
        self.dm.create_main_dir(save_dir_train)
        save_dir_test = os.path.join(main_output_folder,'Test','slices')
        self.dm.create_main_dir(save_dir_test)

        save_path_sliced_spectrograms_train = os.path.join(save_dir_train,"spectrograms_sliced.npy")
        save_path_sliced_spectrograms_test = os.path.join(save_dir_test,"spectrograms_sliced.npy")
        train_size = int(len(data_sliced.spectrograms)*0.9)
        test_size = len(data_sliced.spectrograms)-train_size
        buf_array = []
        for i in range(train_size):
            buf_array.append(data_sliced.spectrograms[i])
        self.save_array_to_numpy_file(buf_array,save_path_sliced_spectrograms_train)
        buf_array.clear()
        for i in range(train_size,train_size+test_size):
            buf_array.append(data_sliced.spectrograms[i])
        self.save_array_to_numpy_file(buf_array,save_path_sliced_spectrograms_test)
        self.save_detail_to_numpy_files(data_sliced,save_dir_train,save_dir_test,train_size,test_size)

    def save_dataset_to_numpy_files(self,dataset_folder,main_output_folder):
        self.dm.create_main_dir(main_output_folder)
        self.dm.create_main_dir(os.path.join(main_output_folder,'Train'))
        self.dm.create_main_dir(os.path.join(main_output_folder,'Test'))

        data_full = self.read_full_spectrograms_to_array(dataset_folder)
        data_sliced = self.read_sliced_spectrograms(dataset_folder)

        self.save_full_spectrograms(main_output_folder,data_full)
        self.save_slice_spectrograms(main_output_folder,data_sliced)

    def read_spectrograms_file(self,train_data_path):
        spectrogram_array = []
        save_path_dims = os.path.join(train_data_path,"spectrograms_dims.npy")
        save_path_spectrograms = os.path.join(train_data_path,"spectrograms.npy")
        size_array = np.load(save_path_dims,allow_pickle=True)
        spect = np.load(save_path_spectrograms,allow_pickle=True)
        for i in range(len(spect)):
            spectrogram_array.append(np.reshape(spect[i],size_array[i]))
        return spectrogram_array

    def read_sliced_spectrograms_file(self,train_data_path):
        save_path = os.path.join(train_data_path,'slices','spectrograms_sliced.npy')
        return np.load(save_path,allow_pickle=True)

    def read_numpy_file(self,train_data_path,name):
        path = os.path.join(train_data_path,name)
        numpy_array = np.load(path,allow_pickle=True)
        return numpy_array

