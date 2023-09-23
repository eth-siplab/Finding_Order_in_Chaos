import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import scipy.signal
from scipy import signal, stats
from scipy.fft import rfft, rfftfreq

sos = signal.butter(4, [0.5, 4], btype='bandpass', fs=64, output='sos')
sos_imu = signal.butter(4, [0.5, 4], btype='bandpass', fs=32, output='sos')
whole_dataset = []
signal_list = []
label_list = []
ppg_fft_list = []
for file in os.listdir():
    d = os.path.join('', file)
    if os.path.isdir(d):
        print(d)
        with open(d + '/' + d + '.pkl', 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            ppg_signal = data['signal']['wrist']['BVP']

            filtered_ppg = signal.sosfilt(sos, ppg_signal)
            segments = np.lib.stride_tricks.sliding_window_view(np.squeeze(filtered_ppg), 64 * 8)[::128]
            z_scored = stats.zscore(segments, axis=1)
            resampled = scipy.signal.resample(z_scored, 200, axis=1)
            labels = data['label']
            signal_list.append(resampled)
            label_list.append(np.expand_dims(labels, 1))
            whole_dataset.append([resampled, np.round(labels)])

dataDalia = dict(whole_dataset=whole_dataset)
with open('Dalia_data.pkl', 'wb') as handle:
    pickle.dump(dataDalia, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('exit')