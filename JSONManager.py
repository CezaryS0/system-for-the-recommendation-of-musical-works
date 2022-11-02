import json

class JSONManager:

    def __init__(self) -> None:
        pass

    def file_open(self,path):
        self.path = path
        self.file = open(path)

    def save_dict_to_JSON(self,dictionary,save_path):
        json_object = json.dumps(dictionary, indent=4)
        self.file = open(save_path, "w")
        self.file.write(json_object)

    def read_JSON(self,path):
        data = json.load(self.file)
        for i in data:
            print(i)
        return data

    def closeFile(self):
        self.file.close()
            