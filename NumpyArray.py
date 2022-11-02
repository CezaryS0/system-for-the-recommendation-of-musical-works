import os 
import cv2
import numpy as np
from TrainingData import TrainingData
from JSONManager import JSONManager

class NumpyArray:

    def __init__(self) -> None:
        pass
    
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
        data = TrainingData()
        data.spectrograms.append(matrix)
        data.ids.append(json_file_data['id'])
        data.titles.append(json_file_data['title'])
        data.samplefreq.append(json_file_data['samplefreq'])
        data.sample_points.append(json_file_data['sample_points'])
        data.tempo_bpm.append(json_file_data['tempo_bmp'])
        data.rolloff_freq.append(json_file_data['tuning'])
        data.duration.append(json_file_data['duration'])
        data.tonic.append(json_file_data['tonic'])
        data.key_sigantures.append(json_file_data['key_signature'])
        data.z_dist_avg_to_tonic.append(json_file_data['z_dist_avg_to_tonic'])

    def read_spectrograms_to_array(self,main_output_folder):
        
        json = JSONManager()
        for folder in os.scandir(main_output_folder):
            if folder.is_dir:
                dirPath = os.path.join(main_output_folder,folder.name)
                cvImage = cv2.imread(os.path.join(dirPath,folder.name+".jpg"))
                matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
                json.file_open(os.path.join(dirPath,folder.name+".json"))
                json_file_data = json.read_JSON()
                self.assingValuesToLists(json_file_data,matrix)
                

    def slices_to_numpy_array(self,main_output_folder):
        slices_array = self.get_array_of_slices()
        numpy_slices = np.array(slices_array)
        return numpy_slices
