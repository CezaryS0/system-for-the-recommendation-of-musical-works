import os 
import cv2
import numpy as np

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

    def read_spectrograms_to_numpy_array(self,main_output_folder):
        spectrograms_array =[]
        for folder in os.scandir(main_output_folder):
            if folder.is_dir:
                cvImage = cv2.imread(os.path.join(main_output_folder,folder.name,folder.name+".jpg"))
                spectrograms_array.append(cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY))
        return spectrograms_array

    def slices_to_numpy_array(self,main_output_folder):
        slices_array = self.get_array_of_slices()
        numpy_slices = np.array(slices_array)
        return numpy_slices
