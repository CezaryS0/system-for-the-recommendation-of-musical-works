import cv2
import numpy as np
class Utils:

    def __init__(self) -> None:
        pass

    def StringToInt(self,string):
        number = 0
        try:
            number = int(string)
        except:
            number = -1
        return number

    def read_image_to_numpy(self,path):
        cvImage = cv2.imread(path,cv2.IMREAD_UNCHANGED)
        matrix = cv2.cvtColor(cvImage,cv2.COLOR_BGR2GRAY)
        return matrix

    def convert_Image_to_cv2(self,image):
        return cv2.cvtColor(np.array(image),cv2.COLOR_BGR2GRAY)