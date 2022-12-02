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
                    matrix = self.ut.read_image_to_numpy(file)
                    self.assingValuesToLists(data,json_file_data,matrix)
                json.closeFile()
        return data

    def flatten_matrix(self,matrix):
        x,y = np.shape(matrix)
        return np.reshape(matrix,x*y),x,y
    
    def expand_and_normalize(self,songs,axis):
        songs = songs.astype(np.float32)
        songs = np.expand_dims(songs,axis=axis)
        songs = songs/255
        return songs

    def expand(self,arr,axis):
        return np.expand_dims(arr,axis)

    def reshape_images(self,data):
        reshaped_list = []
        size_list = []
        for elem in data.spectrograms:
            flat_mat,x,y = self.flatten_matrix(elem) 
            size_list.append([x,y])
            reshaped_list.append(flat_mat)
        return reshaped_list,size_list

    def convert_to_numpy_array(self,matrix):
        if np.isscalar(matrix):
            if type(matrix) is float:
                return np.asarray(matrix,dtype=np.float32)
        else:
            if type(matrix[0]) is float:
                return np.asarray(matrix,dtype=np.float32)
        return np.asarray(matrix)

    def save_array_to_numpy_file(self,arr,path):
        np.save(path,self.convert_to_numpy_array(arr))

    def save_one_feature_to_numpy_file(self,key,data,path_train,path_test):
        save_path_train = os.path.join(path_train,key+'.npy')
        save_path_test = os.path.join(path_test,key+'.npy')
        train_array,test_array = data.split_the_dataset_from_key(key,0.9)
        self.save_array_to_numpy_file(train_array,save_path_train)
        self.save_array_to_numpy_file(test_array,save_path_test)

    def save_detail_to_numpy_files(self,data,path_train,path_test):
        for key in data.dataDict:
            self.save_one_feature_to_numpy_file(key,data,path_train,path_test)

    def save_spectrogram_representations(self,data,model,spectrogram_path):
        spectrograms = self.read_sliced_spectrograms_file(spectrogram_path+'/Train')
        spectrograms =  self.expand_and_normalize(spectrograms,3)
        for image in spectrograms:
            image = np.expand_dims(image,axis=0)
            data.representations_train.append(model.model_predict(image))
        self.save_array_to_numpy_file(data.representations_train,spectrogram_path+'/Train/representations.npy')
        spectrograms = self.read_sliced_spectrograms_file(spectrogram_path+'/Test')
        spectrograms =  self.expand_and_normalize(spectrograms,3)
        for image in spectrograms:
            image = np.expand_dims(image,axis=0)
            data.representations_test.append(model.model_predict(image))
        self.save_array_to_numpy_file(data.representations_test,spectrogram_path+'/Test/representations.npy')

    def save_final_representations(self,model,fusion_path):
        fusion_test = self.numpy.read_numpy_file(fusion_path,'fusion.npy')

    def save_data_fusion(self,data,output_path):
        fusion_train, fusion_test = data.create_data_fusion()
        self.save_array_to_numpy_file(fusion_train,output_path+'/Train/fusion.npy')
        self.save_array_to_numpy_file(fusion_test,output_path+'/Test/fusion.npy')

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
        np.save(save_path_train_spectrograms,np.asarray(buf_array,dtype=object))
        np.save(save_path_train_dims,np.asarray(buf_size_array))
        buf_array.clear()
        buf_size_array.clear()
        for i in range(train_size,train_size+test_size):
            buf_array.append(spectrogram_array[i])
            buf_size_array.append(size_array[i])
        np.save(save_path_test_spectrograms,np.asarray(buf_array,dtype=object))
        np.save(save_path_test_dims,np.asarray(buf_size_array))
        self.save_detail_to_numpy_files(data_full,
        os.path.join(main_output_folder,'Train'),
        os.path.join(main_output_folder,'Test'))
      

    def save_slice_spectrograms(self,main_output_folder,data_sliced:TrainingData):

        save_dir_train = os.path.join(main_output_folder,'Train','slices')
        self.dm.create_main_dir(save_dir_train)
        save_dir_test = os.path.join(main_output_folder,'Test','slices')
        self.dm.create_main_dir(save_dir_test)
        save_path_sliced_spectrograms_train = os.path.join(save_dir_train,"spectrograms_sliced.npy")
        save_path_sliced_spectrograms_test = os.path.join(save_dir_test,"spectrograms_sliced.npy")
        train_array,test_array = data_sliced.split_the_dataset_from_array(data_sliced.spectrograms,0.9)
       
        self.save_array_to_numpy_file(test_array,save_path_sliced_spectrograms_test)
        self.save_array_to_numpy_file(train_array,save_path_sliced_spectrograms_train)
        self.save_detail_to_numpy_files(data_sliced,save_dir_train,save_dir_test)

    def save_dataset_to_numpy_files(self,dataset_folder,main_output_folder):
        self.dm.create_main_dir(main_output_folder)
        self.dm.create_main_dir(os.path.join(main_output_folder,'Train'))
        self.dm.create_main_dir(os.path.join(main_output_folder,'Test'))
        data_sliced = self.read_sliced_spectrograms(dataset_folder)
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

    def append_to_numpy(self,numpy_path,data):
        numpy_array = np.load(numpy_path,allow_pickle=True)
        numpy_array = np.append(numpy_array,data)
        np.save(numpy_path,numpy_array)

    def append_spectrograms_to_numpy(self,full_spect_path,sliced_path,numpy_path,details):
        
        save_path_sliced_spectrograms= os.path.join(numpy_path,'Test','slices',"spectrograms_sliced.npy")
        save_path_spectrograms = os.path.join(numpy_path,'Test',"spectrograms.npy")
        save_path_spectrograms_dims = os.path.join(numpy_path,'Test',"spectrograms.npy")
        matrix = self.ut.read_image_to_numpy(full_spect_path)
        flat_mat,x,y = self.flatten_matrix(matrix)
        self.append_to_numpy(save_path_spectrograms,flat_mat)
        self.append_to_numpy(save_path_spectrograms_dims,[x,y])
        for key in details:
            buf = os.path.join(numpy_path,'Test',key+'.npy')
            self.append_to_numpy(buf,details[key])
        slice_list = self.dm.get_all_files_in_dir(sliced_path)
        matrix_array = []
        for slice in slice_list:
            matrix_array.append(self.ut.read_image_to_numpy(slice))
            for key in details:
                buf = os.path.join(numpy_path,'Test','slices',key+'.npy')
                self.append_to_numpy(buf,details[key])
        self.append_to_numpy(save_path_sliced_spectrograms,matrix_array)