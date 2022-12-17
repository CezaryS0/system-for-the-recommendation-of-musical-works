import sqlite3
import numpy as np
import os
import re
import ast

class Database:

    def __init__(self) -> None:
        pass

    def connect_to_database(self):
        self.connection = sqlite3.connect("Model/Database/aquarium.db")
        self.cursor = self.connection.cursor()

    def array2str(self,arr, precision=None):
        s=np.array_str(arr, precision=precision)
        return s.replace('\n', ',')

    def str2array(self,s):
        # Remove space after [
        s=re.sub('\[ +', '[', s.strip())
        # Replace commas and spaces
        s=re.sub('[,\s]+', ', ', s)
        return np.array(ast.literal_eval(s))

    def create_table(self):
        self.cursor.execute("CREATE TABLE representations (title TEXT, representation TEXT, shape TEXT, dtype TEXT)")
    def insert_into_tables(self,title,matrix:np.ndarray):
        shape = str(np.shape(matrix))
        print(shape)
        numpy_string = np.array2string(matrix.flatten(),separator=", ",threshold=np.inf)
        self.cursor.execute("INSERT INTO {tn} (title, representation, shape,dtype) VALUES(?,?,?,?)".format(tn='representations'),(title, numpy_string,shape,str(matrix.dtype)))

    def drop_database(self):
        if os.path.isfile(os.path.abspath("Model/Database/aquarium.db")):
            os.remove(os.path.abspath("Model/Database/aquarium.db"))

    def read_database(self):
        rows = self.cursor.execute("SELECT * FROM representations")
        title_array = []
        buf_array = []
        for row in rows:
            shape = tuple(map(int, row[2][1:-1].split(', ')))
            type_of_matrix = row[3]
            matrix = np.fromstring(row[1][1:-1],type_of_matrix,sep=", ").reshape(shape)
            title_array.append(row[0])
            buf_array.append(matrix)
        return title_array,buf_array
