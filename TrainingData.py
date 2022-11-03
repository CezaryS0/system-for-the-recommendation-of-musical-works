class TrainingData:

    def __init__(self) -> None:

        self.fields =list()
        self.dataDict = dict()
        
        self.ids = list()
        self.spectrograms = list()
        self.titles =list()
        self.samplefreq = list()
        self.sample_points = list()
        self.tempo_bpm = list()
        self.rolloff_freq = list()
        self.tuning = list()
        self.duration = list()
        self.tonic = list()
        self.key_sigantures = list()
        self.z_dist_avg_to_tonic = list()

    def clear(self):
        self.ids.clear()
        self.spectrograms.clear()
        self.titles.clear()
        self.samplefreq.clear()
        self.sample_points.clear()
        self.tempo_bpm.clear()
        self.rolloff_freq.clear()
        self.tuning.clear()
        self.duration.clear()
        self.tonic.clear()
        self.key_sigantures.clear()
        self.z_dist_avg_to_tonic.clear()