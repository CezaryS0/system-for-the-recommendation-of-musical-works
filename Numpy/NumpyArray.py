import os 
import cv2
import numpy as np
from Numpy.TrainingData import TrainingData
from Managers.JSONManager import JSONManager
from Managers.DirectoryManager import DirectoryManager


class NumpyArray:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
    
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
                cvImage = cv2.imread(os.path.join(dirPath,folder.name+".png"))
                matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
                jsonPath = os.path.join(dirPath,folder.name+".json")
                json.file_open(jsonPath,'r')
                json_file_data = json.read_JSON()
                self.assingValuesToLists(data,json_file_data,matrix)
                json.closeFile()

        return data

    def read_image_to_numpy(self,path):
        cvImage = cv2.imread(path)
        matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
        return matrix

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
                    cvImage = cv2.imread(os.path.join(dirPath,file))
                    matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
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


    def save_full_spectrograms(self,main_output_folder,data_full,):
        save_path_spectrograms = os.path.join(main_output_folder,"spectrograms.npy")
        save_path_dims = os.path.join(main_output_folder,"spectrograms_dims.npy")
        spectrogram_array,size_array = self.reshape_images(data_full)
        np.save(save_path_spectrograms,np.array(spectrogram_array,dtype=object))
        np.save(save_path_dims,np.array(size_array,dtype=object))
        for key in data_full.dataDict:
            save_path = os.path.join(main_output_folder,key+'.npy')
            if type(data_full.dataDict[key][0]) is float:
                np.save(save_path,np.array(data_full.dataDict[key],dtype=np.float32))
            else:
                np.save(save_path,np.array(data_full.dataDict[key],dtype=object))

    def save_slice_spectrograms(self,main_output_folder,data_sliced):
        save_dir = os.path.join(main_output_folder,'slices')
        self.dm.create_main_dir(save_dir)
        save_path_sliced_spectrograms = os.path.join(save_dir,"spectrograms_sliced.npy")
        np.save(save_path_sliced_spectrograms,data_sliced.spectrograms)
        for key in data_sliced.dataDict:
            save_path = os.path.join(save_dir,key+'.npy')
            if type(data_sliced.dataDict[key][0]) is float:
                np.save(save_path,np.array(data_sliced.dataDict[key],dtype=np.float32))
            else:
                np.save(save_path,np.array(data_sliced.dataDict[key],dtype=object))


    def save_dataset_to_numpy_files(self,dataset_folder,main_output_folder):
        self.dm.create_main_dir(main_output_folder)
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
        return np.load(save_path)

    def read_numpy_file(self,train_data_path,name):
        path = os.path.join(train_data_path,name)
        numpy_array = np.load(path,allow_pickle=True)
        return numpy_array

