import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import librosa
import json
import numpy as np
#Define all major scales to be used later for finding key signature
#Arrays all in the format:  [C, C#, D, Eb, E, F, F#, G, Ab, A, Bb, B]
majorscales = {'C' : [1,0,1,0,1,1,0,1,0,1,0,1],
               'C#': [1,1,0,1,0,1,1,0,1,0,1,0],
               'D' : [0,1,1,0,1,0,1,1,0,1,0,1],
               'Eb': [1,0,1,1,0,1,0,1,1,0,1,0],
               'E' : [0,1,0,1,1,0,1,0,1,1,0,1],
               'F' : [1,0,1,0,1,1,0,1,0,1,1,0],
               'F#': [0,1,0,1,0,1,1,0,1,0,1,1],
               'G' : [1,0,1,0,1,0,1,1,0,1,0,1],
               'Ab': [1,1,0,1,0,1,0,1,1,0,1,0],
               'A' : [0,1,1,0,1,0,1,0,1,1,0,1],
               'Bb': [1,0,1,1,0,1,0,1,0,1,1,0],
               'B' : [0,1,0,1,1,0,1,0,1,0,1,1]}


class Audio:

    def __init__(self) -> None:
        self.fileDict = dict()
        self.y=None
        self.sr=None

    def load_file(self,path,csv,index):
        try:
            self.y,self.sr = librosa.load(path,mono=True)
            self.track_title = csv.get_title(index)
            self.tempo, self.beat_frames = librosa.beat.beat_track(y=self.y,sr=self.sr)
            self.tuning = librosa.estimate_tuning(y=self.y,sr=self.sr)
            self.duration = self.y.shape[0]/self.sr
            self.beat_times = librosa.frames_to_time(self.beat_frames,sr=self.sr)
            self.rolloff_freq = round(np.mean(librosa.feature.spectral_rolloff(y=self.y,sr=self.sr,hop_length=512,roll_percent=0.9)),0)
            self.mel = self.generate_spectrogram()
            self.tonic, self.key_signature,self.z_dist_avg_to_tonic = self.findTonicAndKey()
            
        except:
            return
    
    def get_file_details(self,index) ->None:
        
        self.fileDict.update({"id":index})
        self.fileDict.update({"title":self.track_title}) #title_of_a_track
        self.fileDict.update({"samplefreq":self.sr})
        self.fileDict.update({"sample_points":self.y.shape[0]})
        self.fileDict.update({"tempo_bpm":self.tempo}) #tempo of a song in bmps
        self.fileDict.update({"rolloff_freq":self.rolloff_freq}) #Get the rolloff frequency - the frequency at which the loudness drops off by 90%, like a low pass filter
        self.fileDict.update({"tuning":self.tuning}) #tuning
        self.fileDict.update({"duration":self.duration}) #length of a track in seconds
        self.fileDict.update({"tonic":self.tonic})
        self.fileDict.update({"key_signature":self.key_signature})
        self.fileDict.update({"z_dist_avg_to_tonic":self.z_dist_avg_to_tonic})

        return self.fileDict
        
    def save_detail_to_JSON(self,save_path):
        json_object = json.dumps(self.fileDict, indent=4)
        with open(save_path, "w") as outfile:
            outfile.write(json_object)

    def generate_spectrogram(self):
        melspectrogram_array = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128,fmax=8000)
        mel = librosa.power_to_db(melspectrogram_array)
        return mel
    
    def save_spectrogram(self,savepath):
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = float(self.mel.shape[1]) / float(100)
        fig_size[1] = float(self.mel.shape[0]) / float(100)
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis('off')
        plt.axes([0.,0.,1.,1.0],frameon=False,xticks=[],yticks=[])
        librosa.display.specshow(self.mel, cmap='gray_r')
        plt.savefig(savepath, bbox_inches=None, pad_inches=0)
        plt.close()

    def slice_spectrogram(self,spectrogram_path,filename,save_path):
        img = Image.open(spectrogram_path)
        subsample_size = 128
        width, height = img.size
        number_of_samples = int(width / subsample_size)
        for i in range(number_of_samples):
            start = i*subsample_size
            img_temporary = img.crop((start,0.,start+subsample_size,subsample_size))
            img_temporary.save(save_path+'/'+filename+"_"+str(i)+".jpg")

    def findTonicAndKey(self):
        chromagram = librosa.feature.chroma_stft(y=self.y,sr=self.sr)
        chromasums = []
        for i,_ in enumerate(chromagram):
            chromasums.append(np.sum(chromagram[i]))
        tonicval = np.where(max(chromasums)==chromasums)[0][0]
        notes = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        tonic = notes[tonicval]
      
        z_dist_avg_to_tonic = round((max(chromasums)-np.mean(chromasums))/np.std(chromasums), 4)
        
        bestmatch = 0
        bestmatchid = 0
        for key, scale in majorscales.items():
            corr = np.corrcoef(scale, chromasums)[0,1]
            if (corr > bestmatch):
                bestmatch = corr
                bestmatchid = key
        if (tonic != bestmatchid):
            keysig = tonic + ' Minor'
        else:
            keysig = tonic + ' Major'        
        return tonic, keysig, float(z_dist_avg_to_tonic)