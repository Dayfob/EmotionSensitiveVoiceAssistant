import torch.nn as nn
import torch.nn.functional as F
import librosa.feature
import numpy as np
import torch
import librosa
import pyaudio
from faster_whisper import WhisperModel
from collections import defaultdict
import demoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, wordnet
from nltk import pos_tag
from autocorrect import Speller
from transformers import BertTokenizer, BertForSequenceClassification
import time


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


# Function to record audio using pyaudio
# def record_audio(sample_rate):
#     print("Press Enter to start recording...")
#     input()
#     print("Recording... Press Enter again to stop.")
#
#     audio = pyaudio.PyAudio()
#     stream = audio.open(format=FORMAT,
#                         channels=CHANNELS,
#                         rate=sample_rate,
#                         input=True,
#                         frames_per_buffer=FRAME_SIZE)
#                         # frames_per_buffer=CHUNK)
#     frames = []
#     try:
#         while True:
#             # data = stream.read(CHUNK)
#             data = stream.read(FRAME_SIZE)
#             frames.append(data)
#             if input():  # Wait for Enter to stop recording
#                 break
#     except KeyboardInterrupt:
#         pass
#     finally:
#         print("Recording stopped.")
#         stream.stop_stream()
#         stream.close()
#         audio.terminate()
#
#     # Convert recorded frames to a NumPy array
#     audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
#     return audio_data
def record_audio(sample_rate, duration):
    print("Recording...")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)
    frames = []
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            data = stream.read(CHUNK)
            frames.append(data)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Convert recorded frames to a NumPy array
    audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio_data


# Function to load and split audio into segments
def split_audio_into_segments(audio_data, sample_rate, duration):
    # segment_size = sample_rate * duration
    # num_segments = len(audio_data) // segment_size
    # segments = np.array_split(audio_data[:num_segments * segment_size], num_segments)

    num_segments = int(len(audio_data) / (sample_rate * duration))
    segments = np.array_split(audio_data[:num_segments * FRAME_SIZE], num_segments)
    return segments


# Function to classify audio
def classify_audio_segment(segment, sample_rate):
    segment = noise_reduction(segment, sample_rate)
    # Extract features
    extracted_features, mel_spectrogram = extract_features(segment, sample_rate)

    # Prepare input tensors
    features_tensor = torch.tensor(extracted_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    mel_spectrogram_tensor = torch.tensor(mel_spectrogram.reshape(-1, 1, N_MELS, TIME_FRAMES), dtype=torch.float32)

    # Get predictions
    with torch.no_grad():
        output_cnn1 = model1(features_tensor.to(device))
        output_cnn2 = model2(mel_spectrogram_tensor.to(device))

    # Calculate the average probability for each class
    avg_probabilities = (output_cnn1 + output_cnn2) / 2
    avg_probabilities = avg_probabilities * 100
    avg_probabilities = avg_probabilities.flatten()

    max_index = torch.argmax(avg_probabilities.cpu()).item()
    emotion = emotions[max_index]

    return emotion, avg_probabilities.cpu().numpy()


def clean_text(text):
    # Replace emojis
    text = demoji.replace(text)

    # Remove smart quotes and dashes
    text = text.replace("“", "\"").replace("”", "\"").replace("-", " ").replace("'", " ")

    # Lowercase text
    text = text.lower()

    # Tokenize text
    words = word_tokenize(text)
    # print(words)

    # Spelling correction + replace all t with not
    words = ['not' if word == 't' else spell(word) for word in words]

    # Remove stop words and non-alphabetic tokens and punctuation
    words = [word for word in words if word.isalnum() and word not in stop_words or word in ['not', 'no']]

    # Stemming (it's more fast but less accurate alternative to Lemmatization)
    # words = [stemmer.stem(word) for word in words]

    # POS tagging and Lemmatization
    tagged_words = pos_tag(words)

    tag_map = defaultdict(lambda: "n")
    tag_map["N"] = "n"
    tag_map["V"] = "v"
    tag_map["J"] = "a"
    tag_map["R"] = "r"

    words = [lemmatizer.lemmatize(word, pos=tag_map[tag[0]]) for word, tag in tagged_words]

    # Return cleaned words as a single string
    return ' '.join(words)


def classify_text(text):
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Get predictions
    with torch.no_grad():
        outputs = textClassificator(**inputs)
        # probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        probabilities = torch.sigmoid(outputs.logits)

    # Get the predicted labels
    emotion_probabilities = probabilities.flatten().tolist()
    max_prob = max(emotion_probabilities)
    max_idx = emotion_probabilities.index(max_prob)
    predicted_emotion = idx_to_emotion[max_idx]

    print(f"Text: {text}")
    print(f"Predicted Emotion: {predicted_emotion} ({max_prob:.2f})")
    # print("Probabilities for each emotion:")
    # for emotion, prob in zip(emotion_to_idx.keys(), emotion_probabilities):
    #     print(f"{emotion}: {prob:.2f}")

    return predicted_emotion, emotion_probabilities


# SPEECH-EMOTION CLASSIFICATION ====================================================================================
# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the models
model1 = CNN_1().to(device)
model1.load_state_dict(torch.load('speech/outputs/model_CNN_2_1.t'))
model1.eval()

model2 = CNN_2().to(device)
model2.load_state_dict(torch.load('speech/outputs/model_CNN_2_2.t'))
model2.eval()

# Audio configuration
SAMPLE_RATE = 22050  # Hz
DURATION = 1  # seconds
FRAME_SIZE = int(SAMPLE_RATE * DURATION)
N_MELS = 16
TIME_FRAMES = int(704 / N_MELS)
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# SPEECH-TO-TEXT ====================================================================================================

SAMPLE_RATE_WHISPER = 16000
model_size = "medium.en"
whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

# TEXT-EMOTION CLASSIFICATION ====================================================================================

spell = Speller()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('text/outputs/tokenizer_3')
textClassificator = BertForSequenceClassification.from_pretrained('text/outputs/bert_full_model_3', num_labels=7)
textClassificator.eval()  # Set the model to evaluation mode

# List of emotions
emotion_to_idx = {
    'happy': 0,
    'surprised': 1,
    'neutral': 2,
    'sad': 3,
    'fear': 4,
    'angry': 5,
    'disgust': 6
}
idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
MAX_LEN = 128

# Infinite loop for recording and classification
if __name__ == "__main__":
    while True:
        emotion_counter_speech = {
            'happy': 0,
            'surprised': 0,
            'neutral': 0,
            'sad': 0,
            'fear': 0,
            'angry': 0,
            'disgust': 0
        }
        emotion_counter_text = {
            'happy': 0,
            'surprised': 0,
            'neutral': 0,
            'sad': 0,
            'fear': 0,
            'angry': 0,
            'disgust': 0
        }
        user_input = input("Type # of seconds to record audio, or type 'q' to quit: ")
        if user_input.lower() == 'q':
            print("Exiting...")
            break

        audio_data = record_audio(SAMPLE_RATE, user_input)

        recorded_duration = len(audio_data) / SAMPLE_RATE
        print(f"Recorded audio length: {recorded_duration:.2f} seconds")

        # audio_data = audio_data.astype(np.float32) / 32768.0
        audio_data = audio_data.astype(np.float32)
        segments = split_audio_into_segments(audio_data, SAMPLE_RATE, DURATION)

        for i, segment in enumerate(segments):
            emotion, probabilities = classify_audio_segment(segment, SAMPLE_RATE)
            print(f"Segment {i + 1} - Detected Emotion: {emotion}")
            # print(f"Probabilities: {probabilities}")


        # audio_data_whisper = librosa.resample(audio_data, orig_sr=SAMPLE_RATE, target_sr=SAMPLE_RATE_WHISPER)
        # whisperSegments, _ = whisper.transcribe(audio_data_whisper, beam_size=5)
        whisperSegments, _ = whisper.transcribe(audio_data / 32768.0, beam_size=5)

        for segment in whisperSegments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            text = clean_text(segment.text)
            predicted_emotion, emotion_probabilities = classify_text(text)
