from keras.models import Model,load_model
from Model.Numpy.NumpyArray import NumpyArray
from Model.Managers.DataManager import DataManager
from keras.utils import plot_model

class ModelManager:

    def __init__(self) -> None:
        self.model = None
        self.numpy = NumpyArray()
        self.dataManager = DataManager()

    def load_trained_model(self,model_path):
        self.model = load_model(model_path)
        self.model.set_weights(self.model.get_weights())

    def model_summary(self):
        if(self.model!=None):
            print(self.model.summary())
    
    def get_number_of_layers(self):
        if self.model!=None:
            return len(self.model.layers)
        return 0
        
    def discard_layers(self,n):
        new_model = Model(self.model.inputs, self.model.layers[n].output)
        self.model = new_model

    def get_model_shape(self):
        if self.model!=None:
            return self.model.output.shape
        return None

    def generate_plot(self,path):
        if self.model!=None:
            plot_model(self.model, to_file=path,show_shapes=True)

    def model_predict(self,input):
        if self.model!=None:
            return self.model.predict(input,verbose=0)
        return None
