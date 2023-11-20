import os
import pandas as pd 
import mne_bids
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch 

shift_value = 1e13
duration = 3
hop_length = 512  
n_fft = 2048 
sampling_audio = 16000

def get_meg_from_raw(raw, data_meg_chunks):
    x = raw.get_data()
    sampling_freq = raw.info["sfreq"] 
    tot_length_meg = x.shape[1] / sampling_freq
    adj_time = 0
    while (adj_time < tot_length_meg):
        start_stop_seconds = np.array([adj_time, adj_time + duration])
        start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
        raw_selection = raw[:, start_sample:stop_sample][0]
        raw_selection = raw_selection * shift_value
        if (raw_selection.shape[1] < (stop_sample - start_sample)):               
            break
        data_meg_chunks.append(raw_selection)
        adj_time = adj_time + duration
    data_tensor = torch.tensor(data_meg_chunks)
    data_meg_chunks.remove
    return data_tensor

def get_meg_spectrogram(data_tensor):
    data_np = data_tensor.numpy()
    n_frames = (1 + (data_np.shape[2] - n_fft) // hop_length) + n_fft // hop_length
    spectrograms = np.zeros((data_np.shape[0], data_np.shape[1], n_fft // 2 + 1, n_frames))
    for i in range(data_np.shape[0]): 
        for j in range(data_np.shape[1]): 
            d = librosa.stft(data_np[i, j], n_fft=n_fft, hop_length=hop_length)
            spectrograms[i, j] = librosa.amplitude_to_db(np.abs(d), ref=np.max)
    spectrograms_tensor = torch.from_numpy(spectrograms)
    return spectrograms_tensor

def get_audio_spectrogram(audio_path, tot_length_audio, data_audio_chunks):
    adj_time = 0
    n_frames = (1 + (sampling_audio * duration - n_fft) // hop_length) + n_fft // hop_length
    while (adj_time < tot_length_audio):
        start = adj_time
        y, sr = librosa.load(audio_path, sr=sampling_audio, offset=start, duration=duration)
        y_db = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        if (y_db.shape[1] < n_frames):     # TODO --> capire se meglio fare padding o rimuovere
            break
        data_audio_chunks.append(y_db)
        adj_time = start + duration
    audio_tensor = torch.tensor(data_audio_chunks)
    data_audio_chunks.remove
    return audio_tensor


