from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

class GDManager:

    def __init__(self) -> None:
        self.gauth = GoogleAuth()      
        self.drive = GoogleDrive(self.gauth)  

    def print_output(self,filename):
        print("Uploading: ",filename)

    def upload_dir(self,absolute_path,folder_id):
        name = os.path.basename(absolute_path)
        dir_metadata = {
            'title': name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [{'id':folder_id}]
        }
        googleFile  = self.drive.CreateFile(metadata=dir_metadata)
        googleFile.Upload()
        new_id = googleFile.get('id')
        return new_id

    def upload_file(self,absolute_path,folder_id):
        name = os.path.basename(absolute_path)
        file_metadata = {
            'title': name,
            'parents': [{'id':folder_id}]
        }
        gfile = self.drive.CreateFile(metadata=file_metadata)
        gfile.SetContentFile(absolute_path)
        gfile.Upload()

    def upload_rec(self,path,folder_id):
        for files in os.listdir(path):
            absolute_path = os.path.join(os.getcwd(),path,files)
            self.print_output(absolute_path)
            if not os.path.isfile(absolute_path):
                new_id = self.upload_dir(files,folder_id)
                self.upload_rec(absolute_path,new_id)
            else:
                self.upload_file(absolute_path,folder_id)
        
    def upload_directory_recursively(self,path,folder_id):
        new_id = self.upload_dir(path,folder_id)
        self.print_output(os.path.join(os.getcwd(),path))
        self.upload_rec(path,new_id)