import json

class JSONManager:

    def __init__(self) -> None:
        pass

    def file_open(self,path,mode):
        self.path = path
        self.file = open(path,mode=mode)

    def save_dict_to_JSON(self,dictionary):
        json_object = json.dumps(dictionary, indent=4)
        self.file.write(json_object)

    def read_JSON(self):
        data = json.load(self.file)
        return data

    def closeFile(self):
        self.file.close()
            