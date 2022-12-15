from Managers.ModelManager import ModelManager

autoencoder2D = ModelManager()
autoencoder2D.load_trained_model('Autoencoder_Saved/autoencoder.h5')
autoencoder2D.generate_plot('Autoencoder_Saved/autoencoder2D.png')
autoencoder1D = ModelManager()
autoencoder1D.load_trained_model('Autoencoder_Saved/autoencoder_secondary.h5')
autoencoder1D.generate_plot('Autoencoder_Saved/autoencoder1D.png')
