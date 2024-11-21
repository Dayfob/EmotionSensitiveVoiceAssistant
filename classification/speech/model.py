import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa.feature
import numpy as np

class CNN_1(nn.Module):
    def __init__(self, n=32):
        super(CNN_1, self).__init__()
        self.n = n

        self.conv1 = nn.Conv1d(1, self.n, kernel_size=3, padding=1)
        self.conv1_dropout = nn.Dropout1d(p=0.3)
        self.conv2 = nn.Conv1d(self.n, self.n // 2, kernel_size=3, padding=1)
        self.conv2_dropout = nn.Dropout1d(p=0.3)
        self.conv3 = nn.Conv1d(self.n // 2, self.n // 2, kernel_size=3, padding=1)
        self.conv3_dropout = nn.Dropout1d(p=0.3)

        # Calculate the correct input size for the fully connected layer
        self.fc1 = nn.Linear((self.n // 2) * (197 // 8), 32)  # 170 // 8 due to three max_pool1d with kernel_size=2
        self.fc2 = nn.Linear(32, 7)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = F.max_pool1d(torch.tanh(self.conv1(x)), 2)
        out = self.conv1_dropout(out)
        out = F.max_pool1d(torch.tanh(self.conv2(out)), 2)
        out = self.conv2_dropout(out)
        out = F.max_pool1d(torch.tanh(self.conv3(out)), 2)
        out = self.conv3_dropout(out)

        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = torch.tanh(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out

class CNN_2(nn.Module):
    def __init__(self):
        super(CNN_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.dropout_conv1 = nn.Dropout2d(p=0.5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.dropout_conv2 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.dropout_conv3 = nn.Dropout2d(p=0.5)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

        self.fc1 = nn.Linear(128 * 2 * (44 // 2 // 2 // 2),
                             256)  # Adjust based on the output size from feature extractor
        self.fc1_dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc2_dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 7)  # 7 output units for 7 emotions
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.dropout_conv1(torch.relu(self.conv1(x))))
        x = self.pool(self.dropout_conv2(torch.relu(self.conv2(x))))
        x = self.pool(self.dropout_conv3(torch.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1_dropout(torch.relu(self.fc1(x)))
        x = self.fc2_dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return self.sigmoid(x)

def extract_zcr(audio):
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    zcr_stats = np.concatenate([np.mean(zcr.T, axis=0), np.std(zcr.T, axis=0)])
    return zcr_stats


def extract_chroma(audio, sr):
    chroma = librosa.feature.chroma_stft(S=audio, sr=sr)
    chroma_stats = np.concatenate([np.mean(chroma.T, axis=0), np.std(chroma.T, axis=0)])
    return chroma_stats


def extract_mfccs(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_stats = np.concatenate([np.mean(mfccs.T, axis=0), np.std(mfccs.T, axis=0)])
    return mfcc_stats


def extract_spectral_contrast(audio, sr):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    spectral_contrast_stats = np.concatenate(
        [np.mean(spectral_contrast.T, axis=0), np.std(spectral_contrast.T, axis=0)])
    return spectral_contrast_stats


def extract_spectral_rolloff(audio, sr):
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_rolloff_stats = np.concatenate([np.mean(spectral_rolloff.T, axis=0), np.std(spectral_rolloff.T, axis=0)])
    return spectral_rolloff_stats


def extract_rmse(audio):
    rmse = librosa.feature.rms(y=audio)
    rmse_stats = np.concatenate([np.mean(rmse.T, axis=0), np.std(rmse.T, axis=0)])
    return rmse_stats


def extract_mel_spectrogram(audio, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spectrogram_mean = np.mean(mel_spectrogram.T, axis=0)
    mel_spectrogram_db = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=16), ref=np.max)
    return mel_spectrogram_db, mel_spectrogram_mean


def extract_features(data, sample_rate):
    result = np.array([])

    # ZCR
    zcr = extract_zcr(data)
    result = np.hstack((result, zcr))

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = extract_chroma(stft, sample_rate)
    result = np.hstack((result, chroma_stft))

    # MFCC
    mfcc = extract_mfccs(data, sample_rate)
    result = np.hstack((result, mfcc))

    # Spectral_contrast
    spectral_contrast = extract_spectral_contrast(data, sample_rate)
    result = np.hstack((result, spectral_contrast))

    # Spectral_rolloff
    spectral_rolloff = extract_spectral_rolloff(data, sample_rate)
    result = np.hstack((result, spectral_rolloff))

    # RMS
    rms = extract_rmse(data)
    result = np.hstack((result, rms))

    # Mel spectrogram
    mel_spectrogram, mel_spectrogram_mean = extract_mel_spectrogram(data, sample_rate)
    result = np.hstack((result, mel_spectrogram_mean))
    return result, mel_spectrogram.flatten()


def get_features(path):
    data, sample_rate = librosa.load(path)

    extracted_features, mel_spectrogram = extract_features(data, sample_rate)
    result = np.array(extracted_features)
    mel_spectrogram = np.array(mel_spectrogram)

    return result, mel_spectrogram

def noise_reduction(audio, sr, noise_profile=None):
    # Compute the STFT of the signal
    stft = librosa.stft(audio)
    magnitude, phase = librosa.magphase(stft)

    # Estimate the noise profile if not provided
    if noise_profile is None:
        noise_profile = np.median(magnitude, axis=1, keepdims=True)

    # Subtract the noise profile from the magnitude
    magnitude_cleaned = np.maximum(magnitude - noise_profile, 0)

    # Reconstruct the audio signal from the cleaned magnitude and original phase
    stft_cleaned = magnitude_cleaned * phase
    audio_cleaned = librosa.istft(stft_cleaned)

    return audio_cleaned