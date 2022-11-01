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