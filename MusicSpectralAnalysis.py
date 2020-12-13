import turicreate as tc
import IPython.display as ipd
import librosa
import numpy as np
from pathlib import Path
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import scipy
%matplotlib inline
from IPython.display import Audio

audio_data = 'Sound/Data/sansa.mp3'




#Load the audio file in librosa

z , sr_z = librosa.load(audio_data, offset = 8.0, duration = 6.0)
z
sr_z

# Verify length of the audio
print('Length of Audio:', np.shape(x)[0]/sr, "seconds")


Audio(data = z, rate = sr_z)

# Use HPSS to get harmoic and percussive components

z_harmonic, z_percussive = librosa.effects.hpss(z)

Audio(data = z_harmonic, rate = sr_z)

Audio(data = z_percussive, rate = sr_z)


tempo, beat_frames = librosa.beat.beat_track(y=z, sr=sr_z)
tempo
beat_frames


beat_times = librosa.frames_to_time(beat_frames, sr=sr_z)
beat_times

def plot_wave(signal, sampling_rate):
    plt.figure(figsize=(20, 10))
    librosa.display.waveplot(signal, sr=sampling_rate)

plot_wave(z_harmonic, sr_z)



z_fft = scipy.fftpack.fft(z_harmonic, sr_z)

def fft_plt(z_fft):
    len_z_fft = len(z_fft)
    xf = np.linspace(0.0, sr/(2.0), int(len_z_fft/2))
    plt.figure(figsize=(10, 6))
    plt.plot(xf, 2.0/len_z_fft * np.abs(z_fft[:len_z_fft//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.xlim(0, 1500)
    plt.show()

fft_plt(z_fft)



def plt_spect(signal):
    Z = librosa.stft(signal)
    Zdb = librosa.amplitude_to_db(abs(Z))
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(Zdb, sr=sr_z, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')


plt_spect(z_percussive)
plt_spect(z_harmonic)

plt_spect(z)

# Let's plot spectrograms again

#Extract stft
window = 2048
slider_length = 512

#This is STFT which is in complex domain
z_stft = librosa.stft(z, n_fft = window, hop_length = slider_length)
z_stft.shape
type(z_stft[0][0])

#Real domain from the complex domain
z_real = np.abs(z_stft) ** 2
z_real.shape
type(z_real[0][0])

def plot_spectrogram(signal, sampling_rate, slider_length, y_axis="linear"):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(signal, sr=sampling_rate, hop_length=slider_length, x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")


plot_spectrogram(z_real, sr_z, slider_length)



# Log amplitde spectogram because above simple amplitude didn't give much info. We percieve sounds logarithmically.
# So, we need to change the amplitude from linear to logarithmic

z_real_log = librosa.power_to_db(z_real)
plot_spectrogram(z_real_log, sr_z, slider_length)

#Log frequency spectogram
#Above is quite squished, so let's transform the frequency logarithmically too


plot_spectrogram(z_real_log, sr_z, slider_length, y_axis="log")

# C2 to C4 are 65 to 262 Hz
# G6 to A6 are 1568 to 1760 Hz
# Frequencies at lower Frequency can be easily distinuighsed by humans but not the higher ones
# Humans percieve sound logarithmically not linearly
# Mel Spectpgrams present the perceptualy relevant frequency representation
# Mel scale is logarithmic in nature


#Mel spectogram
filter_banks = librosa.filters.mel(n_fft=2048, sr=22050, n_mels=10)
filter_banks.shape

def mel_filter_bank():
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(filter_banks,
                             sr=sr_z,
                             x_axis="linear")
    plt.colorbar(format="%+2.f")
    plt.show()

mel_filter_bank()

n_fft = 2048
hop_length = 256
n_mels = 5
mel_spect = librosa.feature.melspectrogram(z, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
log_mel_spect= librosa.power_to_db(mel_spect)

def mel_grams():
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(log_mel_spect,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format="%+2.f dB")
    plt.show()

mel_grams()
