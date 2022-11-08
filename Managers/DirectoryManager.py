import os

class DirectoryManager:

    def __init__(self) -> None:
        pass
    
    def create_main_dir(self,main_output_folder):
        if not os.path.exists(main_output_folder):
            os.makedirs(main_output_folder)

    def create_filename_dir(self,current_path,filename):
        path_buf_array = os.path.join(current_path,filename)
        os.makedirs(path_buf_array)
        os.makedirs(os.path.join(path_buf_array,'slices'))
        return path_buf_array

    def get_all_files_in_dir(self,dir_path) -> list:
        file_list = []
        for file in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path,file)):
                file_list.append(file)
        return file_list

    def create_dirs(self,root,dataset_path,main_output_folder) -> str:     
        current_path = root.lstrip(dataset_path+'\\')
        output_path = ""
        if not current_path=='':
            print(current_path)
            path_array = current_path.split('\\')
            path_buf_array = main_output_folder+'\\'
            for p in path_array:
                path_buf_array = os.path.join(path_buf_array,p)
                if not os.path.exists(path_buf_array):
                    print(path_buf_array)
                    os.makedirs(path_buf_array)
            output_path = path_buf_array
        return output_path
