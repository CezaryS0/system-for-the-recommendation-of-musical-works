import tkinter as tk
from tkinter import *
from Numpy.NumpyArray import NumpyArray
from Numpy.NumpyArray import np
from Controller.Controller import Controller
from tkinter import filedialog as fd
import os
from threading import Thread
os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
import vlc

class Window:

    def __init__(self) -> None:
        self.window = tk.Tk()
        self.window.geometry("800x500")
        self.window.title("System for the recommendation of musical works")
        self.numpy = NumpyArray()
        self.p =vlc.MediaPlayer()
        self.controller = Controller()
        self.currentTrack = ""
        self.thread = Thread()
        self.listbox = None

    def create_frame(self,master):
        canvas = tk.Frame(master,bg="gray")
        canvas.pack(fill=tk.BOTH,expand=True)
        return canvas

    def create_background(self,master):
        img = PhotoImage(file = 'View/assets/background.gif')
        label1 = Label(self.window, image = img)
        label1.image_names=(img)
        label1.place(x = 0,y = 0)
        label1.pack()

    def create_label(self,w,h,text,master,fill):

        label = tk.Label(
            master=master,
            text=text,
            fg="white",
            bg="black",
            font=("Arial", 25),
            width=w,
            height=h
        )
        label.pack(fill=fill,expand=False)
        return label

    def playCallback(self):
        if not self.currentTrack == "":
            if not self.p.is_playing():
                self.p = vlc.MediaPlayer(self.currentTrack)
                self.p.play()

    def stopCallback(self):
        if self.p.is_playing():
           self.p.stop()

    def loadCallback(self):
        self.currentTrack = fd.askopenfilename()
        self.song_title.config(text=os.path.basename(self.currentTrack))
    
    def recommendation_thread(self):
        list_array = self.controller.generate_recommendations(self.currentTrack)
        self.recommendation_label.config(text=list_array[0][0])

    def recommendCallback(self):
        if not self.currentTrack == "":
            if not self.thread.is_alive():
                self.thread = Thread(target = self.recommendation_thread)
                self.thread.start()
        
    def create_scrollable_list(self,arr,master):
        self.listbox = Listbox(self.window)
        self.listbox.pack(side = LEFT, fill = Y)
        scrollbar = Scrollbar(self.window)
        scrollbar.pack(side = LEFT, fill = Y)
        for title in arr:
             self.listbox.insert(END, title)
        self.listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.listbox.yview)

    def create_window(self):
        self.create_label(100,0,"Music Recommender",self.window,tk.X)
        title_array = np.unique(self.numpy.read_numpy_file('Train_Data/Test/slices','title.npy'))
        self.create_scrollable_list(title_array,self.window)
        frame = self.create_frame(self.window)
        frame2 = tk.Frame(master=frame,bg="gray")
        frame2.pack(side=tk.BOTTOM,anchor=S)
        frame3 = tk.Frame(master=frame)
        frame3.pack(side=TOP,fill=X, anchor=NW,expand=True)
        current_track = Label(frame3,text="CurrentTrack",font=("Arial", 25))
        current_track.pack(side=tk.TOP)
        self.song_title = Label(frame3,text="Not Selected",font=("Arial", 25),pady=10)
        self.song_title.pack(side=tk.TOP)
        recommendation_word = Label(frame3,text="Recommendation: ",font=("Arial", 25),pady=10)
        recommendation_word.pack(side=tk.TOP,anchor=NW)
        self.recommendation_label = Label(frame3,text="",font=("Arial", 25))
        self.recommendation_label.pack(side=tk.TOP,anchor=NW)

        button = tk.Button(master=frame2,text='Play',bg='#0052cc',font=('Helvetica',15), fg='#ffffff',command=self.playCallback)
        button.pack(side=tk.LEFT,anchor=S,padx=(10,10),pady=(0,10))
        button = tk.Button(master=frame2,text='Stop',bg='#0052cc', fg='#ffffff',font=('Helvetica',15),command=self.stopCallback)
        button.pack(side=tk.LEFT,anchor=S,padx=(0,10),pady=(0,10))
        button = tk.Button(master=frame2,text='Load file',bg='#0052cc',font=('Helvetica',15), fg='#ffffff',command=self.loadCallback)
        button.pack(side=tk.LEFT,anchor=S,padx=(0,10),pady=(0,10))
        button = tk.Button(master=frame2,text='Recommend',bg='#0052cc',font=('Helvetica',15), fg='#ffffff',command=self.recommendCallback)
        button.pack(side=tk.LEFT,anchor=S,padx=(0,10),pady=(0,10))

        self.window.mainloop()
        