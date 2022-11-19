from Managers.ModelManager import ModelManager

model = ModelManager()
model.load_trained_model('Saved_Model/Model.h5')
model.discard_layers(-2)
model.generate_plot('Saved_Model/Architecture.png')
model.model_summary()