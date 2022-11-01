from itertools import count
import os
from DirectoryManager import DirectoryManager
from CSVManager import CSVManager
from Utils import Utils
from Audio import Audio

class DataManager:

    def __init__(self) -> None:
        pass

    def create_and_slice_spectrograms(self,main_output_folder,dataset_path,csv_path):
        dm = DirectoryManager()
        csv = CSVManager(csv_path)
        ut = Utils()
        audio = Audio()
        tracks_array = csv.read_data_from_csv()
        tracks_id_array = tracks_array[:, 0]
        tracks_id_list = list(tracks_id_array.reshape(tracks_id_array.shape[0], 1))
        counter = 0
        dm.create_main_dir(main_output_folder)
        for root, _, files in os.walk(dataset_path):
            for f in files:
                filename, file_ext = os.path.splitext(f)
                current_track_id = ut.StringToInt(filename)
                if file_ext.upper() == ".MP3":
                    index = tracks_id_list.index(current_track_id)
                    filename_folder_path = dm.create_filename_dir(main_output_folder,filename)
                    full_path = os.path.join(root,f)
                    savepath = os.path.join(filename_folder_path,filename+".jpg")
                    audio.load_file(full_path,csv,index)
                    audio.save_spectrogram(savepath)
                    audio.slice_spectrogram(savepath,filename,os.path.join(filename_folder_path,'slices'))
                    audio.get_file_details()
                    audio.save_detail_to_JSON(filename_folder_path+'/'+filename+'.json')
                    counter+=1
                    if counter==1:
                        return

