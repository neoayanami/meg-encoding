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
sampling_meg = 1000
freq_cut = 30
num_channel = 208     

'''
 I task si riferiscono alle 4 storie:
 --> task 0: lw1
 --> task 1: cable_spool_fort
 --> task 2: easy_money
 --> task 3: the_black_willow
'''

meg_path = '/data01/data/MEG'
patient = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
session = ['0', '1']
task = {'lw1': 0.0, 'cable_spool_fort': 1.0, 'easy_money': 2.0, 'the_black_willow': 3.0}
lw1 = ['0.0', '1.0', '2.0', '3.0']
cable_spool_fort = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0']
easy_money = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0']
the_black_willow = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0', '7.0', '8.0', '9.0', '10.0', '11.0']


def get_bids_raw(meg_path, subject, session, task):
    bids_path = mne_bids.BIDSPath(
        subject = subject,
        session = session, 
        task = task, 
        datatype = "meg", 
        root = meg_path,
    )
    try:
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)
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


def get_epochs(raw, story_uid, sound_id):
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0
    meta=meta[(meta["kind"]=="word") & (meta["story_uid"]==story_uid) & 
              (meta["sound_id"]==sound_id)]  
    # if meta.empty:
        # raise ValueError(f"No matching meta entries found for story_uid: {story_uid} and sound_id: {sound_id}")

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


def get_meg_spectrogram_ranged(data_tensor, f_min, f_max):
    data_np = data_tensor.numpy()
    n_frames = (1 + (data_np.shape[2] - n_fft) // hop_length) + n_fft // hop_length
    n_freq_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sampling_meg // 2, n_freq_bins)
    f_min_idx = np.searchsorted(freqs, f_min)
    f_max_idx = np.searchsorted(freqs, f_max, side='right')
    selected_bins = f_max_idx - f_min_idx
    spectrograms = np.zeros((data_np.shape[0], data_np.shape[1], selected_bins, n_frames))
    for i in range(data_np.shape[0]): 
        for j in range(data_np.shape[1]): 
            d = librosa.stft(data_np[i, j], n_fft=n_fft, hop_length=hop_length)
            db_spectrogram = librosa.amplitude_to_db(np.abs(d), ref=np.max)
            spectrograms[i, j] = db_spectrogram[f_min_idx:f_max_idx]
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


def get_audio_mel_spectrogram(audio_path, epochs, n_mels=128):
    data_audio_chunks = []
    n_frames = (1 + ((sampling_audio * duration) - n_fft) // hop_length) + n_fft // hop_length
    epoch_spectr = get_meg_from_raw_epochs(epochs)
    for i in range(epoch_spectr.shape[0]):
        start = epochs[i]._metadata["start"].item()
        y, sr = librosa.load(audio_path, sr=sampling_audio, offset=start, duration=duration) 
        # extract Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, 
                                                         hop_length=hop_length, n_mels=n_mels)
        mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)
        if (mel_spectrogram_db.shape[1] < n_frames):
            # Make padding
            pad_width = n_frames - mel_spectrogram_db.shape[1]
            mel_spectrogram_db = np.pad(mel_spectrogram_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        data_audio_chunks.append(mel_spectrogram_db)
    audio_tensor = torch.tensor(data_audio_chunks)
    return audio_tensor


def plot_spectrogram(encode_spect, sr, sample, channel):
    if (len(encode_spect.shape) == 4):
        time_extent = np.linspace(0, 3.21, encode_spect.shape[1])
        encode_spect =  encode_spect[sample][channel].numpy()
        if (encode_spect.shape[0] != n_fft // 2 + 1): 
            freq_extent = np.linspace(0, freq_cut, encode_spect.shape[0])
            plt.figure(figsize=(10, 4))
            plt.imshow(encode_spect, aspect='auto', origin='lower', 
                        extent=[time_extent.min(), time_extent.max(), freq_extent.min(), freq_extent.max()])
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'MEG Spectrogram (Sample {sample}, Channel {channel})')
        else:    
            freq_extent = np.linspace(0, n_fft, encode_spect.shape[0])
            plt.imshow(encode_spect, aspect='auto', origin='lower', 
                        extent=[time_extent.min(), time_extent.max(), freq_extent.min(), freq_extent.max()])
            plt.colorbar(format='%+2.0f dB')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'MEG Spectrogram (Sample {sample}, Channel {channel})')
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(encode_spect, 
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


def save_data(data, flag, patient, session, task, audio_name, audio_snip):
    save_dir = os.path.join(meg_path, 'collect_data', flag)
    save_path = os.path.join(save_dir, f'{patient}_{session}_{task}_{audio_name}_{audio_snip}.pt')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_path):
        torch.save(data, save_path)
    else: 
        print(f"File {save_path} already exists.")
    


