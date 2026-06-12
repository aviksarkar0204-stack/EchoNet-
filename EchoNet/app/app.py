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


def load_model():
    model = models.efficientnet_b0(weights=None)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, NUM_CLASSES)
    )
    model.load_state_dict(torch.load('../models/echonet_efficientnet.pth', map_location='cpu'))
    model.eval()
    return model


model = load_model()
le = joblib.load('../models/label_encoder.pkl')


def predict(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                       n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.mean()) / (S_db.std() + 1e-8)

    tensor = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().numpy()

    top5_idx = probs.argsort()[-5:][::-1]
    top5 = {CLASSES[i]: float(probs[i]) for i in top5_idx}

    return top5


demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type='filepath', label='Upload Audio Clip'),
    outputs=gr.Label(num_top_classes=5, label='Predicted Sound'),
    title='🎧 EchoNet — Environmental Sound Classifier',
    description='Upload a .wav audio clip and EchoNet will classify it into one of 50 environmental sound categories.',
    examples=[]
)

demo.launch()