# system-for-the-recommendation-of-musical-works 
<br>
System for the recommendation of musical works by Cezary Szumerowski

## General overwiev of the project

The purpose of this project was to try to implemented a recommendation engine for the musical works using the content-based filtering.
<br><br>
https://en.wikipedia.org/wiki/Recommender_system#Content-based_filtering<br>

## Main idea

### The implementation utilizes

1. MFCC spectrogrograms.
2. Two convolutional autoencoders.
    1. Two-dimensional (Conv2D).
    2. One-dimensional (Conv1D). 
3. Cosine similarity.

Autoencoders were trained on randomly picked tracks from the full FMA dataset.<br>
https://os.unil.cloud.switch.ch/fma/fma_full.zip

### Recommendation process

Each selected song is encoded using the first two-dimensional autoencoder, which generates a small matrix representation of each track.<br>
Next, each mathematical representation is flattened and concatenated to the non-subjective features of a track consisting of:
1. Key of the song
2. Roll-off frequency
3. Tempo of the track (BPM)
<br>

This will create a **data fusion** of each track.<br><br>
Later on, each generated fusion is encoded using the second one-dimensional autoencoder, which generates **final representations** of each track.
Every generated vector is saved in a SQLITE3 Database, for making recommendations later, when a new track comes in.<br>
When the new track appears (The Anchor Song) it goes through the encoding and the fusion generation process, which leads to the generation of its **final representation**.<br>

Next, using the cosine similarity, the distance between the anchor song and each song from the database is calculated.<br>
The song with the smallest distance from the anchor song (or the highest simialrity) becomes the recommendation.<br>

## Detailed description

### MFCC Spectrograms

The mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency, while The Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal spectrum. This frequency warping can allow for better representation of sound, for example, in audio compression that might potentially reduce the transmission bandwidth and the storage requirements of audio signals.<br>
https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

**Example spectrogram:**

![068234_cropped](https://user-images.githubusercontent.com/56163226/218075379-ce66f58c-2859-4243-8227-84d41ec4c883.png)

### Autoencoders 2D and 1D
<p align="center">
  <button>
    <img src="https://user-images.githubusercontent.com/56163226/218076324-7c0c0194-118b-41ca-9832-ea45138fe23c.png" height="1800" title="Lol"/>
   </button>
   <button>
    <img src="https://user-images.githubusercontent.com/56163226/218076478-fc963000-ff8c-4403-bf71-182a4f422a96.png" height="1800"/>
   </button>
</p>

The two-dimensional Autoencoder is build of 2 parts:<br>

1. Encoder part
2. Decoder part

The encoder part consists of 6 convolutional layers, the same as the decoder part but instead of the 6 MaxPooling2D layers used in the encoder the decoder uses UpSampling2D layers to recontruct the encoded image.<br>

The one-dimensional Autoencoder is build of 2 parts:<br>

1. Encoder part
2. Decoder part

The encoder part consists of 6 convolutional layers, the same as the decoder part but instead of the 6 MaxPooling1D layers used in the encoder the decoder uses UpSampling1D layers to recontruct the encoded image. At the output there is also a Cropping1D layer, due to the data fusion input image size not being a power of two.<br>

### Cosine similarity

<p align="center">
  <img src="https://user-images.githubusercontent.com/56163226/218095960-2fc18892-8192-45f4-92de-8849ff08a27b.jpg"/>
</p><br>


**Cosine similarity** is a measure of similarity between two non-zero vectors defined in an inner product space. Cosine similarity is the cosine of the angle between the vectors; that is, it is the dot product of the vectors divided by the product of their lengths. It follows that the cosine similarity does not depend on the magnitudes of the vectors, but only on their angle. The cosine similarity always belongs to the interval [-1,1].<br>


