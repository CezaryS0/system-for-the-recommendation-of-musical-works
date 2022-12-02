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

    def search_folder(self,inputId):
        filelist = self.drive.ListFile({'q': "'{}' in parents and trashed=false".format(inputId)}).GetList()
        return filelist
    
    def delete_files(self,file_list,absolute_path):
        name = os.path.basename(absolute_path)    
        for file in file_list:
            if file['title'] == name:
                gfile = self.drive.CreateFile({'id':file['id']})
                gfile.Delete()
    
    def delete_file(self,file_id):
        gfile = self.drive.CreateFile({'id':file_id})
        gfile.Delete()

    def upload_file(self,absolute_path,folder_id):
        name = os.path.basename(absolute_path)
        file_metadata = {
            'title': name,
            'parents': [{'id':folder_id}]
        }
        self.print_output(absolute_path)
        gfile = self.drive.CreateFile(metadata=file_metadata)
        gfile.SetContentFile(absolute_path)
        gfile.Upload()

    def get_folder_id_by_name(self,name,folder_id):
        new_id = 0
        file_list = self.search_folder(folder_id)
        for file in file_list:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                if file['title'] == name:
                    new_id = file['id']
                    break
        return new_id

    def get_file_id_by_name(self,name,folder_id):
        new_id = 0
        file_list = self.search_folder(folder_id)
        for file in file_list:
            if file['title'] == name:
                new_id = file['id']
                break
        return new_id

    def upload_file_to_folder(self,name,absolute_path,folder_id):
        new_id = self.find_folder_id_rec(name,folder_id)
        print(new_id)
        file_id = self.get_file_id_by_name('representations.npy',new_id)
        if file_id!=0:
            self.delete_file(file_id)
        self.upload_file(absolute_path,new_id)

    def find_folder_id_rec(self,name,folder_id):
        file_list = self.search_folder(folder_id)
        new_id =0
        for file in file_list:
            if file['mimeType'] == 'application/vnd.google-apps.folder':
                if file['title'] == name:
                    print(file['id'])
                    new_id = file['id']
                    break
                else:
                    new_id = self.find_folder_id_rec(name,file['id'])
        return new_id

    def upload_rec(self,path,folder_id):
        for files in os.listdir(path):
            absolute_path = os.path.join(os.getcwd(),path,files)
            self.print_output(absolute_path)
            if not os.path.isfile(absolute_path):
                new_id = self.upload_dir(files,folder_id)
                self.upload_rec(absolute_path,new_id)
            else:
                self.upload_file(absolute_path,folder_id)
        
    def overwrite_directory(self,path,folder_id,overwrite):
        new_id = 0
        if overwrite==True:
            file_list = self.search_folder(folder_id)
            self.delete_files(file_list,path)
            new_id = self.upload_dir(path,folder_id)
        else:
            new_id = self.get_folder_id_by_name(path,folder_id)
        return new_id

    def upload_directory_recursively(self,path,folder_id,overwrite):
        new_id = self.overwrite_directory(path,folder_id,overwrite)
        if new_id!=0:
            self.print_output(os.path.join(os.getcwd(),path))
            self.upload_rec(path,new_id)