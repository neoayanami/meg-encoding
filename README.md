# spect-to-meg

The main goal is to make an audio/brain encoding model.
The input will be the audio and the output will be some representation of the MEG signal.

Step 1: Get familiar with the mne library to extract MEG data and have corresponding MEG and audio signals;

Step 2: Create a representation of the audio and MEG, (e.g., Mel Spectrogram and stft or wavelet, etc.);

Step 3: Split the data into train/test;

Step 4: Build a model using a CNN on the Mel Spectrograms. Below is a list of alternatives:
        - Deep Mel Latent Space
        - Wav2Vec pretrained embeddings
        - Train an autoencoder and use the latent representation as input for our encoding model.
        Each of these spaces can be used as input and the goal is to estimate (i.e., regression) the corresponding MEG representation;

Step 5: Validation, measuring metrics and hyperparameter optimization (define some metrics of interest);

Step 6: MEG Analysis. We will choose a basic MEG signal analysis that we will do both on the real data and the predicted data;

Step 7: Hyperparameter optimization, baselines and research questions. Surely it will be necessary to show both our method and the baselines from the literature and/or proposals by us;

Step 8: Multimodal encoding.
