import pandas as pd

class CSVManager:

    def __init__(self,csv_path) -> None:
        self.csv_file = csv_path

    def read_data_from_csv(self):
        tracks = pd.read_csv(self.csv_file,header=2,low_memory=False)
        tracks_array = tracks.values
        return tracks_array
    
    def get_title(self,index):
        tracks_array = self.read_data_from_csv()
        tracks_title_list = list(tracks_array[:, 52])
        return tracks_title_list[index]