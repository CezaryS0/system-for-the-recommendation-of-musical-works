import tkinter as tk
import tkinter.ttk as ttk
from Numpy.NumpyArray import NumpyArray
from Managers.DataManager import DataManager
from Controller.Controller import Controller
from tkinter import filedialog as fd
from View.RoundedButton import RoundedButton
from View.Widget import Widget
import os
from threading import Thread
os.add_dll_directory(r'C:\Program Files\VideoLAN\VLC')
import vlc
from operator import itemgetter

class Window:

    def __init__(self) -> None:
        self.window = tk.Tk()
        self.style = ttk.Style()
        self.window.geometry("1150x600")
        self.window.minsize(1150,600)
        self.window.title("System for the recommendation of musical works")
        self.numpy = NumpyArray()
        self.p =vlc.MediaPlayer()
        self.dm = DataManager()
        self.widget = Widget()
        self.controller = Controller()
        self.currentTrack = ""
        self.thread = Thread()
        self.listbox = None

    def create_frame(self,master):
        canvas = tk.Frame(master,bg="white")
        canvas.pack(fill=tk.BOTH,expand=True)
        return canvas

    def create_background(self,master):
        img = tk.PhotoImage(file = 'View/assets/background.gif')
        label1 = tk.Label(self.window, image = img)
        label1.image_names=(img)
        label1.place(x = 0,y = 0)
        label1.pack()

    def create_widget(self,master,text):

        frame1 = ttk.Frame(master,style="RoundedFrame", padding=10)
        text1 = tk.Text(frame1, borderwidth=0, highlightthickness=0, wrap="word",width=60, height=4,font=('Times',20))
        text1.pack(fill="both",side="top", expand=False)
        text1.bind("<FocusIn>", lambda event: frame1.state(["focus"]))
        text1.bind("<FocusOut>", lambda event: frame1.state(["!focus"]))
        text1.delete(0.0,"end")
        text1.insert(0.0, text)
        frame1.pack(side="top", fill="both", expand=True, padx=20, pady=20)
        return text1

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
        text=os.path.basename(self.currentTrack)
        self.textBox1.delete(0.0,"end")
        self.textBox1.insert("end",'Current Track\n\n' + text)
    
    def recommendation_thread(self):
        list_array = self.controller.generate_recommendations(self.currentTrack)
        self.textBox2.delete(0.0,"end")
        self.textBox2.insert("end",'Recommendation\n\n' +'Song: '+ list_array[0][1])

    def recommendCallback(self):
        if not self.currentTrack == "":
            if not self.thread.is_alive():
                self.thread = Thread(target = self.recommendation_thread)
                self.thread.start()

    def onselect(self,evt):
        w = evt.widget
        index = int(w.curselection()[0])
        self.currentTrack =  self.dm.get_track_by_ID('Spectrograms',self.test_list[index][0])
        text=os.path.basename(self.currentTrack)
        self.textBox1.delete(0.0,"end")
        self.textBox1.insert("end",'Current Track\n\n' + text)

    def create_scrollable_list(self,n):
        self.listbox = tk.Listbox(self.window,width=n)
        self.listbox.bind('<<ListboxSelect>>', self.onselect)
        self.listbox.pack(side = tk.LEFT, fill = tk.Y)
        scrollbar = tk.Scrollbar(self.window)
        scrollbar.pack(side = tk.LEFT, fill = tk.Y)
        for title in self.test_list:
             self.listbox.insert(tk.END, title[1])
        self.listbox.config(yscrollcommand = scrollbar.set)
        scrollbar.config(command = self.listbox.yview)

    def load_test_songs(self):
        title_array = self.numpy.read_numpy_file('Train_Data/Test/slices','title.npy')
        id_array = self.numpy.read_numpy_file('Train_Data/Test/slices','id.npy')
        test_files_dict = dict()
        for i in range(len(id_array)):
            if not id_array[i] in test_files_dict:
                test_files_dict.update({id_array[i]:title_array[i]})
        self.test_list = list(test_files_dict.items())
        self.test_list = sorted(self.test_list,key=itemgetter(1))
        return title_array

    def create_buttons(self,frame2):
        btn = RoundedButton(master=frame2,text="Play", radius=50, btnbackground="#0078ff", btnforeground="#ffffff",clicked=self.playCallback)
        btn.pack(side=tk.LEFT,anchor=tk.S,padx=(10,10),pady=(0,10))
        btn = RoundedButton(master=frame2,text="Stop", radius=50, btnbackground="#0078ff", btnforeground="#ffffff",clicked=self.stopCallback)
        btn.pack(side=tk.LEFT,anchor=tk.S,padx=(10,10),pady=(0,10))
        btn = RoundedButton(master=frame2,text="Load", radius=50, btnbackground="#0078ff", btnforeground="#ffffff",clicked=self.loadCallback)
        btn.pack(side=tk.LEFT,anchor=tk.S,padx=(10,10),pady=(0,10))
        btn = RoundedButton(master=frame2,text="Recommend", radius=50, btnbackground="#0078ff", btnforeground="#ffffff",clicked=self.recommendCallback)
        btn.pack(side=tk.LEFT,anchor=tk.S,padx=(10,10),pady=(0,10))

    def create_window(self):
        borderImage = tk.PhotoImage("borderImage", data=self.widget.borderImageData)
        focusBorderImage = tk.PhotoImage("focusBorderImage", data=self.widget.focusBorderImageData)
        self.style.element_create("RoundedFrame",
                     "image", borderImage,
                     ("focus", focusBorderImage),
                     border=16, sticky="nsew")
        self.style.layout("RoundedFrame",[("RoundedFrame", {"sticky": "nsew"})])
        self.create_label(100,0,"🎵Music Recommender🎵",self.window,tk.X)
        title_array = self.load_test_songs()
        self.create_scrollable_list(len(max(title_array, key=len)))
        frame = self.create_frame(self.window)
        frame2 = tk.Frame(master=frame,bg="white")
        frame2.pack(side=tk.BOTTOM,anchor=tk.S,expand=True)
        self.textBox1 = self.create_widget(frame,"Current Track\n")
        self.textBox2 = self.create_widget(frame,"Recommendation\n")
        self.create_buttons(frame2)
        self.window.mainloop()
        