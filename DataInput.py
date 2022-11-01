

class DataInput:

    label_list_index = {
        "tempo": 0,
        "tuning": 1,
        "duration" :2,
        
    }
    

    def __init__(self) -> None:
        self.images = []
        self.labels_list = list()