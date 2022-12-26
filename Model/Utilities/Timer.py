import time
import os
class Timer:

    def __init__(self) -> None:
        self.st = 0.0
        self.et = 0.0
        self.elapsed_time = 0.0

    def startTimer(self):
        self.st = time.time()

    def endTimer(self):
        self.et = time.time()
        self.elapsed_time = self.et - self.st

    def saveResults(self,label,path,overwrite):
        if os.path.isfile(path) and overwrite==True:
            os.remove(path)
        with open(path,'a+') as file:
            res = self.elapsed_time*1000
            file.write(label+' : '+str(res)+' miliseconds\n\n')
