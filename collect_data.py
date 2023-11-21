import os
import pandas as pd 
import mne_bids
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch 

shift_value = 1e13
duration = 3   # secondsß
decim = 1
n_fft = 512 
hop_length = n_fft // 4  
sampling_audio = 16000

# maybe to remove
def get_meg_from_raw_toRm(raw, data_meg_chunks):
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

def get_bids_raw(meg_path, subject, session, task):
    bids_path = mne_bids.BIDSPath(
        subject = subject,
        session = session, 
        task = task, 
        datatype = "meg", 
        root = meg_path,
    )
    try:
        raw = mne_bids.read_raw_bids(bids_path)
    except FileNotFoundError:
        print("missing", subject, session, task)
        pass
    raw = raw.pick_types(
        meg=True, misc=False, eeg=False, eog=False, ecg=False
    )
    return raw

# TODO: applicare lo shift value
def get_meg_from_raw_epochs(epochs):
    data_meg = epochs.get_data()
    tensor_meg = torch.tensor(data_meg)
    return tensor_meg

def get_epochs(raw, sound_id):
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    meta=meta[(meta["kind"]=="word") & (meta["sound_id"]==sound_id)]  
    # events/epochs
    events = np.c_[meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))].astype(int)
    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=duration,      # time_window hyperparam
        decim=decim,        # how many points define the temporal window
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )
    # threshold
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs

def get_meg_spectrogram(data_tensor):
    data_np = data_tensor.numpy()
    n_frames = (1 + (data_np.shape[2] - n_fft) // hop_length) + n_fft // hop_length
    spectrograms = np.zeros((data_np.shape[0], data_np.shape[1], n_fft // 2 + 1, n_frames))
    for i in range(data_np.shape[0]): 
        for j in range(data_np.shape[1]): 
            d = librosa.stft(data_np[i, j], n_fft=n_fft, hop_length=hop_length)
            # spectrograms[i, j] = librosa.amplitude_to_db(np.abs(d), ref=np.max)
            spectrograms[i, j] = d
    spectrograms_tensor = torch.from_numpy(spectrograms)
    return spectrograms_tensor

# maybe to remove
def get_audio_spectrogram_toRm(audio_path, tot_length_audio, data_audio_chunks):
    adj_time = 0
    n_frames = (1 + (sampling_audio * duration - n_fft) // hop_length) + n_fft // hop_length
    while (adj_time < tot_length_audio):
        start = adj_time
        y, sr = librosa.load(audio_path, sr=sampling_audio, offset=start, duration=duration)
        y_db = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
        if (y_db.shape[1] < n_frames):     
            break
        data_audio_chunks.append(y_db)
        adj_time = start + duration
    audio_tensor = torch.tensor(data_audio_chunks)
    data_audio_chunks.remove
    return audio_tensor

def get_audio_spectrogram(audio_path, epochs):
    data_audio_chunks = []
    n_frames = (1 + ((sampling_audio * duration) - n_fft) // hop_length) + n_fft // hop_length
    epoch_spectr = get_meg_from_raw_epochs(epochs)
    for i in range(epoch_spectr.shape[0]):
        start = epochs[i]._metadata["start"].item()
        y, sr = librosa.load(audio_path, sr=sampling_audio, offset=start, duration=duration)
        y_db = librosa.amplitude_to_db(librosa.stft(y, n_fft=n_fft, hop_length=hop_length), ref=np.max)
        if (y_db.shape[1] < n_frames):   
            # make padding         
            pad_width = n_frames - y_db.shape[1]
            y_db = np.pad(y_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        data_audio_chunks.append(y_db)
    audio_tensor = torch.tensor(data_audio_chunks)
    return audio_tensor


