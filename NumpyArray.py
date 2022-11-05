import os 
import cv2
import numpy as np
from soupsieve import select
from TrainingData import TrainingData
from JSONManager import JSONManager
from DirectoryManager import DirectoryManager


class NumpyArray:

    def __init__(self) -> None:
        self.data = TrainingData()
        self.dm = DirectoryManager()
    
    def get_array_of_slices(self,main_output_folder):
        slices_array = []
        for folder in os.listdir(main_output_folder):
            if os.path.isdir(folder):
                slices_path = os.path.join(folder,'slices')
                for slice in os.listdir(slices_path):
                    if os.path.isfile(slice):
                        _, file_ext = os.path.splitext(slice)
                        if file_ext.upper() == ".JPG":
                            cvImage = cv2.imread(slice,cv2.IMREAD_UNCHANGED)
                            slices_array.append(cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY))
        return slices_array 


    def assingValuesToLists(self,json_file_data,matrix):
        
        for name in json_file_data:
            if name not in self.data.dataDict:
                self.data.dataDict[name]=list()
                self.data.fields.append(name)
            self.data.dataDict[name].append(json_file_data[name])
        self.data.spectrograms.append(matrix)
        #self.data.ids.append(json_file_data['id'])
        #self.data.titles.append(json_file_data['title'])
        #self.data.samplefreq.append(json_file_data['samplefreq'])
        #self.data.sample_points.append(json_file_data['sample_points'])
        #self.data.tempo_bpm.append(json_file_data['tempo_bmp'])
        #self.data.rolloff_freq.append(json_file_data['tuning'])
        #self.data.duration.append(json_file_data['duration'])
        #self.data.tonic.append(json_file_data['tonic'])
        #self.data.key_sigantures.append(json_file_data['key_signature'])
        #self.data.z_dist_avg_to_tonic.append(json_file_data['z_dist_avg_to_tonic'])

    def read_data_to_array(self,main_output_folder):
        self.data.clear()
        json = JSONManager()
        for folder in os.scandir(main_output_folder):
            if folder.is_dir:
                dirPath = os.path.join(main_output_folder,folder.name)
                cvImage = cv2.imread(os.path.join(dirPath,folder.name+".jpg"))
                matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
                jsonPath = os.path.join(dirPath,folder.name+".json")
                json.file_open(jsonPath,'r')
                json_file_data = json.read_JSON()
                self.assingValuesToLists(json_file_data,matrix)
                json.closeFile()

        return self.data

    def slices_to_numpy_array(self,main_output_folder):
        slices_array = self.get_array_of_slices()
        numpy_slices = np.array(slices_array)
        return numpy_slices


    def reshape_images(self):
        reshaped_list = list()
        for elem in self.data.spectrograms:
            x,y = np.shape(elem)
            reshaped_list.append(np.reshape(elem,x*y))
        return reshaped_list

    def save_dataset_to_numpy_files(self,dataset_folder,main_output_folder):
        self.dm.create_main_dir(main_output_folder)
        self.read_data_to_array(dataset_folder)
        save_path = os.path.join(main_output_folder,"spectrograms.npy")
        spectrogram_array = self.reshape_images()
        np.save(save_path,spectrogram_array)
        for key in self.data.dataDict:
            save_path = os.path.join(main_output_folder,key+'.npy')
            np.save(save_path,np.asarray(self.data.dataDict[key]))
