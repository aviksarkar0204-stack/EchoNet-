import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import joblib
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

N_MELS = 128
SR = 22050
DURATION = 5
N_FFT = 2048
HOP_LENGTH = 512
NUM_CLASSES = 50

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASSES = [
    'airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn',
    'cat', 'chainsaw', 'chirping_birds', 'church_bells', 'clapping',
    'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire',
    'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks',
    'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks',
    'footsteps', 'frog', 'glass_breaking', 'hand_saw', 'helicopter',
    'hen', 'insects', 'keyboard_typing', 'laughing', 'mouse_click',
    'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep',
    'siren', 'sneezing', 'snoring', 'thunderstorm', 'toilet_flush',
    'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind'
]

def load_efficientnet():
    model = models.efficientnet_b0(weights=None)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, NUM_CLASSES)
    )
    model.load_state_dict(torch.load(
        os.path.join(BASE_DIR, 'models', 'echonet_efficientnet.pth'),
        map_location='cpu',
        weights_only=False
    ))
    model.eval()
    return model

def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean = np.mean(centroid)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    return np.hstack([mfcc_mean, mfcc_std, chroma_mean, centroid_mean, zcr_mean])

efficientnet = load_efficientnet()
svm = joblib.load(os.path.join(BASE_DIR, 'models', 'svm_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'models', 'scaler.pkl'))
le = joblib.load(os.path.join(BASE_DIR, 'models', 'label_encoder.pkl'))

def predict(audio_path, model_choice):
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))

    if model_choice == 'EfficientNet (74.5%)':
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
        tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            outputs = efficientnet(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
        top5_idx = probs.argsort()[-5:][::-1]
        return {CLASSES[i]: float(probs[i]) for i in top5_idx}

    else:
        features = extract_features(y, sr).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probs = svm.predict_proba(features_scaled)[0]
        top5_idx = probs.argsort()[-5:][::-1]
        return {le.classes_[i]: float(probs[i]) for i in top5_idx}

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type='filepath', label='Upload Audio Clip'),
        gr.Dropdown(
            choices=['EfficientNet (74.5%)', 'SVM (64.75%)'],
            value='EfficientNet (74.5%)',
            label='Choose Model'
        )
    ],
    outputs=gr.Label(num_top_classes=5, label='Predicted Sound'),
    title='🎧 EchoNet — Environmental Sound Classifier',
    description='Upload a .wav audio clip and choose a model. EchoNet classifies it into one of 50 environmental sound categories.',
)

demo.launch()