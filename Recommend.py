from Managers.ModelManager import ModelManager


name = '000002'
mn = ModelManager()
mn.load_trained_model('Saved_Model/Model.h5')
mn.model_predict(name)
