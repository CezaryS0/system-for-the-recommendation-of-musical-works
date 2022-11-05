import os
from DirectoryManager import DirectoryManager
from CSVManager import CSVManager
from Utils import Utils
from Audio import Audio
from JSONManager import JSONManager
from NumpyArray import NumpyArray

class DataManager:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.ut = Utils()
        self.audio = Audio()
        self.json = JSONManager()
        self.numpy = NumpyArray()

    def save_details_to_Json(self,counter,jsonPath):
        fileDict = self.audio.get_file_details(counter)
        
        self.json.file_open(jsonPath,'w')
        self.json.save_dict_to_JSON(fileDict)
        self.json.closeFile()

    def loadMP3Files(self,root,f,index,filename_folder_path):
        filename, _ = os.path.splitext(f)
        
        full_path = os.path.join(root,f)
        savepath = os.path.join(filename_folder_path,filename+".jpg")
        self.audio.load_file(full_path,self.csv,index)
        self.audio.save_spectrogram(savepath)
        self.audio.slice_spectrogram(savepath,filename,os.path.join(filename_folder_path,'slices'))

    def getTrackIDList(self):
        tracks_array = self.csv.read_data_from_csv()
        tracks_id_array = tracks_array[:, 0]
        tracks_id_list = list(tracks_id_array.reshape(tracks_id_array.shape[0], 1))
        return tracks_id_list

    def create_and_slice_spectrograms(self,main_output_folder,dataset_path,csv_path):
        self.csv = CSVManager(csv_path)
        self.main_output_folder = main_output_folder
        tracks_id_list = self.getTrackIDList()
        counter = 0
        self.dm.create_main_dir(main_output_folder)
        for root, _, files in os.walk(dataset_path):
            for f in files:
                filename, file_ext = os.path.splitext(f)
                current_track_id = self.ut.StringToInt(filename)
                if file_ext.upper() == ".MP3":
                    index = tracks_id_list.index(current_track_id)
                    filename_folder_path = self.dm.create_filename_dir(self.main_output_folder,filename)
                    self.loadMP3Files(root,f,index,filename_folder_path)
                    jsonPath = os.path.join(filename_folder_path ,filename+'.json')
                    self.save_details_to_Json(counter,jsonPath)
                    counter+=1
                    #if counter==1:
                       #return
        
