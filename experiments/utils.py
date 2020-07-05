class speedArray:
    def __init__(self):
        self.__arr = []
        self.arr_len = 0

    def readArr(self, fname):
        self.__arr = [float(l[:-1]) for l in open(fname, "r").readlines()]
        self.arr_len = len(self.__arr)

    def setArr(self, arr):
        self.__arr = arr
        self.arr_len = len(arr)
    
    def get(self, idx):
        if idx < self.arr_len:
            return str(self.__arr[idx])
        return "--"

def openSpeedArr(path, suffix="_predicted"):
    fname = ".".join(path.split(".")[:-1])+"{}.txt".format(suffix)
    speed_array = speedArray()
    try:
        speed_array.readArr(fname)
    except:
        pass
    return speed_array
    