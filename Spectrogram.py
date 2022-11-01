import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image

class Spectrogram:

    def __init__(self) -> None:
        pass

    
    
    def generate_spectrogram(self,y,sr,full_path,savepath) -> None:
        y,sr = librosa.load(full_path)
        melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
        mel = librosa.power_to_db(melspectrogram_array)
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = float(mel.shape[1]) / float(100)
        fig_size[1] = float(mel.shape[0]) / float(100)
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis('off')
        plt.axes([0.,0.,1.,1.0],frameon=False,xticks=[],yticks=[])
        librosa.display.specshow(mel, cmap='gray_r')
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