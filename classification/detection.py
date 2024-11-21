import os
from faster_whisper import WhisperModel
from speech.model import *

audio_path = "../data/speech/3.mp3"
# audio_path = "../data/speech/7.mp3"
# audio_path = "../data/speech/9.mp3"
# audio_path = "../data/speech/12.mp3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Speech emotion classification ======================================================================================

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model1 = CNN_1().to(device)
model1.load_state_dict(torch.load('speech/outputs/model_CNN_2_1.t'))
model1.eval()
model2 = CNN_2().to(device)
model2.load_state_dict(torch.load('speech/outputs/model_CNN_2_2.t'))
model2.eval()

import torch
import numpy as np
import librosa

# Audio configuration
SAMPLE_RATE = 22050  # Hz
DURATION = 1  # seconds
FRAME_SIZE = int(SAMPLE_RATE * DURATION)
N_MELS = 16
TIME_FRAMES = int(704 / N_MELS)
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]


# Function to load audio file and split into 1-second segments
def load_audio_segments(file_path, sample_rate, duration):
    audio_data, _ = librosa.load(file_path, sr=sample_rate)
    num_segments = int(len(audio_data) / (sample_rate * duration))
    segments = np.array_split(audio_data[:num_segments * FRAME_SIZE], num_segments)
    return segments


# # Load the model (assuming model1 and model2 are already defined and loaded)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model1.to(device)
# model2.to(device)

# Load audio file and split into segments
file_path = audio_path  # Replace with your file path
audio_segments = load_audio_segments(file_path, SAMPLE_RATE, DURATION)

# Process each segment for emotion prediction
for i, segment in enumerate(audio_segments):
    # Noise reduction (optional)
    segment = noise_reduction(segment, SAMPLE_RATE)

    # Extract features
    extracted_features, mel_spectrogram = extract_features(segment, SAMPLE_RATE)

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

    # Print or save the results for this segment
    print(f"Time: {i} - {i + 1} sec, Emotion: {emotion}, Probabilities: {avg_probabilities.cpu().numpy()}")


# Text preprocessing ========================================================================================
from collections import defaultdict
import demoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer, wordnet
from nltk import pos_tag
from autocorrect import Speller

# Initialize tools
spell = Speller()
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


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

# Text emotion classification ===================================================================================

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('text/outputs/tokenizer_2')
textClassificator = BertForSequenceClassification.from_pretrained('text/outputs/bert_full_model_2', num_labels=7)
# model.load_state_dict(torch.load('bert_model_2.pth', map_location=torch.device('cpu')))
textClassificator.eval()  # Set the model to evaluation mode

# List of emotions
emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
MAX_LEN = 64

def classify_text(text):
    # Tokenize and prepare input
    inputs = tokenizer(
        text,
        # return_tensors="pt", truncation=True, padding=True,
                       add_special_tokens=True,
                       max_length=MAX_LEN,
                       return_token_type_ids=False,
                       padding='max_length',
                       truncation=True,
                       return_attention_mask=True,
                       return_tensors='pt',
                       )

    print("Inputs: ", inputs)


    # Get predictions
    with torch.no_grad():
        outputs = textClassificator(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        # probabilities = torch.sigmoid(outputs.logits)

    print("Logits??:", outputs)
    # Logits: tensor([[ -37.9633,   -6.7012,  -13.4319,  -39.3405,   -5.2107,  -28.2893, -137.3368]])

    # Get the emotion with the highest probability
    predicted_emotion_index = torch.argmax(probabilities, dim=1).item()
    predicted_emotion = emotions[predicted_emotion_index]
    emotion_probabilities = probabilities.flatten().tolist()

    print(f"Text: {text}")
    print(f"Predicted Emotion: {predicted_emotion}")
    print("Probabilities for each emotion:")
    for emotion, prob in zip(emotions, emotion_probabilities):
        print(f"{emotion}: {prob:.2f}")

    return predicted_emotion, emotion_probabilities


# Example usage
# text = "I'm feeling fantastic today!"
# classify_text(text)


# Speech to text ===================================================================================================

model_size = "medium.en"

# Run on GPU with FP16
whisper = WhisperModel(model_size, device="cpu", compute_type="int8")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = whisper.transcribe(audio_path, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     text = clean_text(segment.text)
#     classify_text(text)


text = "work computer tech ability hyper focus one issue real asset however living day day get bogged feel frustrate not make progress focus one problem"
# text = "I am so happy. Really really happy"
# text = "I am so sad. Really really sad"
print(text)
text = clean_text(text)
classify_text(text)