import os
from Model.Managers.DirectoryManager import DirectoryManager
from Model.Managers.CSVManager import CSVManager
from Model.Utilities.Utils import Utils
from Model.Audio.Audio import Audio
from Model.Managers.JSONManager import JSONManager
from Model.Numpy.NumpyArray import NumpyArray

class DataManager:

    def __init__(self) -> None:
        self.dm = DirectoryManager()
        self.ut = Utils()
        self.audio = Audio()
        self.json = JSONManager()
        self.numpy = NumpyArray()

    def save_details_to_Json(self,fileDict,jsonPath):
        self.json.file_open(jsonPath,'w')
        self.json.save_dict_to_JSON(fileDict)
        self.json.closeFile()

    def get_track_name_from_json(self,jsonPath):
        self.json.file_open(jsonPath,'r')
        data = self.json.read_JSON()
        self.json.closeFile()
        return data['title']

    def get_track_by_ID(self,dataset_path,id):
        for root, _, files in os.walk(dataset_path):
            for f in files:
                _, file_ext = os.path.splitext(f)
                if file_ext.upper() == ".JSON":
                    self.json.file_open(os.path.join(root,f),'r')
                    data = self.json.read_JSON()
                    self.json.closeFile()
                    if data['id'] == id:
                        print(data['filename'])
                        return self.dm.get_file_path_by_name(data['filename'],'dataset/fma_full')
        return None

    def createSpectrograms(self,f,filename_folder_path,n_samples):
        filename, _ = os.path.splitext(f)
        savepath = os.path.join(filename_folder_path,filename+".png")
        slices_path = os.path.join(filename_folder_path,'slices')
        self.audio.save_spectrogtram_mfcc(savepath)
        self.audio.slice_spectrogram(savepath,filename,slices_path,n_samples)
        return savepath,slices_path

    def get_last_index(self,dataset_path):
        subfolders = [ f.path for f in os.scandir(dataset_path) if f.is_dir() ]
        amount = len(subfolders)
        if amount==0:
            return 0
        return amount-1

    def getTrackIDList(self):
        tracks_array = self.csv.read_data_from_csv()
        tracks_id_array = tracks_array[:, 0]
        tracks_id_list = list(tracks_id_array.reshape(tracks_id_array.shape[0], 1))
        return tracks_id_list

    def get_index(self,tracks_id_list,current_track_id):
        index=-1
        try:
            index = tracks_id_list.index(current_track_id)
        except:
            pass
        return index

    def create_dirs_and_save_spectrograms(self,f,filename,counter,n_samples):
        filename_folder_path = self.dm.create_filename_dir(self.main_output_folder,filename)
        self.createSpectrograms(f,filename_folder_path,n_samples)
        jsonPath = os.path.join(filename_folder_path ,filename+'.json')
        fileDict = self.audio.get_file_details(f,counter)
        self.save_details_to_Json(fileDict,jsonPath)

    def create_and_slice_spectrograms(self,main_output_folder,dataset_path,csv_path,n_samples):
        self.csv = CSVManager(csv_path)
        self.main_output_folder = main_output_folder
        tracks_id_list = self.getTrackIDList()
        counter = 0
        self.dm.create_main_dir(main_output_folder)
        for root, _, files in os.walk(dataset_path):
            for f in files:
                filename, file_ext = os.path.splitext(f)
                current_track_id = self.ut.StringToInt(filename)
                if file_ext.upper() == ".WAV" or file_ext.upper() == ".MP3":
                    index = self.get_index(tracks_id_list,current_track_id)
                    if not os.path.exists(os.path.join(self.main_output_folder,filename)):
                        if index>=0:
                            if self.audio.load_file_csv(os.path.join(root,f),self.csv,index,n_samples) == True:
                                self.create_dirs_and_save_spectrograms(f,filename,counter,n_samples)
                                counter+=1
                                print(f)
                        else:
                            if self.audio.load_file(os.path.join(root,f)) == True:
                                self.create_dirs_and_save_spectrograms(f,filename,counter,n_samples)
                                counter+=1
                                print(f)

    def get_test_spectrograms_slices_and_titles(self,dataset_path):
        spectrograms_array = self.numpy.read_numpy_file(dataset_path,'spectrograms_sliced.npy')
        titles_array = self.numpy.read_numpy_file(dataset_path,'title.npy')
        return spectrograms_array,titles_array

    def get_spectrogram_from_name(self,dataset_path,name):
        for folder in os.listdir(dataset_path):
            if not os.path.isfile(folder) and folder.upper() == name.upper():
                dirPath = os.path.join(dataset_path,folder,'slices')
                slice_array = []
                for slice in os.listdir(dirPath):
                    slice_path = os.path.join(dirPath,slice)
                    slice_array.append(self.ut.read_image_to_numpy(slice_path))
                return slice_array
        return None

    def add_song_to_test_dataset(self,main_output_folder,numpy_dir_path,path):
        if os.path.exists(path):
            if self.audio.load_file(path) == True:
                filename = self.dm.get_file_name(path)
                print(filename[0])
                filename_folder_path = self.dm.create_filename_dir(main_output_folder,filename[0])
                full_spect,sliced_path =self.createSpectrograms(filename[0],filename_folder_path)
                jsonPath = os.path.join(filename_folder_path ,filename[0]+'.json')
                index = self.get_last_index(main_output_folder)
                fileDict = self.audio.get_file_details(index)
                self.save_details_to_Json(fileDict,jsonPath)
                self.numpy.append_spectrograms_to_numpy(full_spect,sliced_path,numpy_dir_path,fileDict)