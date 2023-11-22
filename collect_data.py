import os
import pandas as pd 
import mne_bids
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch 

shift_value = 1e13
duration = 3   # seconds
decim = 1
n_fft = 512 
hop_length = n_fft // 4  
sampling_audio = 16000


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
        raw.load_data().filter(0.5, 30.0, n_jobs=1) 
        raw = raw.pick_types(
            meg=True, misc=False, eeg=False, eog=False, ecg=False
        )
    except FileNotFoundError:
        print("missing", subject, session, task)
        pass
    return raw


def get_meg_from_raw_epochs(epochs):
    data_meg = epochs.get_data()
    data_meg = data_meg * shift_value
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
            spectrograms[i, j] = librosa.amplitude_to_db(np.abs(d), ref=np.max)
    spectrograms_tensor = torch.from_numpy(spectrograms)
    return spectrograms_tensor


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


def plot_spectrogram(encode_spect, sr, sample, channel):
    plt.figure(figsize=(10, 4))
    if (len(encode_spect.shape) == 4):
        librosa.display.specshow(encode_spect[sample][channel].numpy(), 
                                 sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                 x_axis='time', y_axis='log')
        plt.title('MEG Spectrogram')
    else:
        librosa.display.specshow(encode_spect[sample].numpy(), 
                                 sr=sr, n_fft=n_fft, hop_length=hop_length, 
                                 x_axis='time', y_axis='log')
        plt.title('Audio Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()
    


