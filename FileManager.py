import csv
import os
import librosa
from CSVManager import CSVManager
import json

class FileManager:

    def __init__(self) -> None:
        self.fileDict = dict()
        self.y=None
        self.sr=None

    def load_file(self,path):
        try:
            self.y,self.sr = librosa.load(path)
        except:
            return None,None
        return self.y,self.sr
    
    def get_file_details(self,csv,index) ->None:
        track_title = csv.get_title(index)
        tempo, beats = librosa.beat.beat_track(self.y,self.sr)
        tuning = librosa.estimate_tuning(self.y,self.sr)
        duration = librosa.get_duration(self.y,self.sr)
        self.fileDict.update({"title":track_title})
        self.fileDict.update({"tempo":tempo})
        self.fileDict.update({"tuning":tuning})
        self.fileDict.update({"duration":duration})

    def save_detail_to_JSON(self,save_path):
        json_object = json.dumps(self.fileDict, indent=4)
        with open(save_path, "w") as outfile:
            outfile.write(json_object)