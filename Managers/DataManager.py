import os
from Managers.DirectoryManager import DirectoryManager
from Managers.CSVManager import CSVManager
from Utils import Utils
from Audio.Audio import Audio
from Managers.JSONManager import JSONManager
from Numpy.NumpyArray import NumpyArray
from pydub import AudioSegment

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

    def get_track_name_from_json(self,jsonPath):
        self.json.file_open(jsonPath,'w')
        data = self.json.read_JSON()
        self.json.closeFile()
        return data['title']

    def createSpectrograms(self,f,filename_folder_path):
        filename, _ = os.path.splitext(f)
        savepath = os.path.join(filename_folder_path,filename+".png")
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
                if file_ext.upper() == ".WAV":
                    index = tracks_id_list.index(current_track_id)
                    if self.audio.load_file(os.path.join(root,f),self.csv,index) == True:
                        filename_folder_path = self.dm.create_filename_dir(self.main_output_folder,filename)
                        self.createSpectrograms(f,filename_folder_path)
                        jsonPath = os.path.join(filename_folder_path ,filename+'.json')
                        self.save_details_to_Json(counter,jsonPath)
                        counter+=1
                        print(f)
                    #if counter==1:
                       #return
    
    def get_spectrograms_and_titles(self,dataset_path):
        for folder in os.listdir(dataset_path):
            if not os.path.isfile(folder):
                jsonPath = os.path.join(dataset_path,folder)
                dirPath = os.path.join(dataset_path,folder,'slices')
                slice_array = []
                title_array = []
                for slice in os.listdir(dirPath):
                    title = self.get_track_name_from_json(os.path.join(jsonPath,folder+'.json'))
                    slice_path = os.path.join(dirPath,slice)
                    slice_array.append(self.numpy.read_image_to_numpy(slice_path))
                    title_array.append(title)
                return slice_array,title_array
        return None

    def get_spectrogram_from_name(self,dataset_path,name):
        for folder in os.listdir(dataset_path):
            if not os.path.isfile(folder) and folder.upper() == name.upper():
                dirPath = os.path.join(dataset_path,folder,'slices')
                slice_array = []
                for slice in os.listdir(dirPath):
                    slice_path = os.path.join(dirPath,slice)
                    slice_array.append(self.numpy.read_image_to_numpy(slice_path))
                return slice_array
        return None
    