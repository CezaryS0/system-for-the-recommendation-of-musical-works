from pydub import AudioSegment
from Managers.DirectoryManager import DirectoryManager
import os

class Converter:

    def __init__(self) -> None:
        self.dm = DirectoryManager()

    def audio(self,save_path,mp3_path):
        try:
            audio = AudioSegment.from_file(mp3_path)
            filename = os.path.split(mp3_path)[1]
            filename = os.path.splitext(filename)[0]
            wav_file_name = os.path.join(save_path,filename+'.wav')
            audio.export(wav_file_name , format="wav")
            return True
        except:
            pass
        return False

    def convertMP3toWAV(self,dataset_path):
        self.dm.create_main_dir('dataset_wav')
        for root, _, files in os.walk(dataset_path):
            for f in files:
                filename,file_ext = os.path.splitext(f)
                if file_ext.upper() == ".MP3":
                    if not os.path.isfile(os.path.join('dataset_wav',filename+'.wav')):
                        print("Converting: ",f)
                        self.audio('dataset_wav',os.path.join(root,f))

dataset = 'dataset/fma_full'
conv = Converter()
conv.convertMP3toWAV(dataset)