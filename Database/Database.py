import sqlite3
import numpy as np
import os


class Database:

    def __init__(self) -> None:
        self.drop_database()
        self.connection = sqlite3.connect("Database/aquarium.db")
        self.cursor = self.connection.cursor()

    def create_table(self):
        self.cursor.execute("CREATE TABLE representations (title TEXT, representation TEXT)")

    def insert_into_tables(self,title,matrix:np.ndarray):
        numpy_string = np.array2string(matrix,separator=',',suppress_small=False)
        self.cursor.execute("INSERT INTO {tn} (title, representation) VALUES(?, ?)".format(tn='representations'),(title, numpy_string))

    def drop_database(self):
        if os.path.isfile(os.path.abspath("Database/aquarium.db")):
            os.remove(os.path.abspath("Database/aquarium.db"))

    def read_database(self):
        rows = self.cursor.execute("SELECT title, representation FROM representations").fetchall()
        return [[row[0],row[1]] for row in rows]