# 🎧 EchoNet — Environmental Sound Classification

A complete audio machine learning pipeline that classifies 50 environmental sounds using classical ML and deep learning. Built on the ESC-50 dataset.

---

## 🏆 Results

| Model | Accuracy |
|---|---|
| SVM + 13 MFCC means | 51.75% |
| SVM + 40 rich features | **64.75%** |
| Custom CNN (from scratch) | 46.00% |
| EfficientNet (transfer learning) | **74.50%** ✅ |

EfficientNet beats the classical ML baseline by **9.75%**.

---

## 📁 Project Structure

```
EchoNet/
│
├── data/
│   └── ESC-50-master/              ← ESC-50 dataset (not pushed to GitHub)
│       ├── audio/                  ← 2,000 .wav clips (5 seconds each)
│       └── meta/
│           └── esc50.csv           ← labels, folds, class names
│
├── notebooks/
│   ├── 01_exploration.ipynb        ← EDA: waveforms, spectrograms, MFCCs
│   ├── 02_classical_ml.ipynb       ← MFCC features + SVM/RF/XGBoost
│   └── 03_cnn.ipynb                ← Mel Spectrogram + CNN + EfficientNet
│
├── src/
│   ├── features.py                 ← MFCC / Mel Spectrogram extraction
│   ├── dataset.py                  ← PyTorch Dataset class
│   └── model.py                    ← CNN architecture
│
├── models/
│   ├── svm_model.pkl               ← trained SVM model
│   ├── scaler.pkl                  ← StandardScaler for SVM features
│   ├── label_encoder.pkl           ← LabelEncoder (number → class name)
│   └── echonet_efficientnet.pth    ← fine-tuned EfficientNet weights
│
├── app/
│   └── app.py                      ← Gradio deployment app
│
├── requirements.txt
└── README.md
```

---

## 🧠 What This Project Does

EchoNet takes a raw audio clip and predicts which of 50 environmental sound categories it belongs to — dogs, rain, helicopters, chainsaw, thunderstorm, and 45 more.

It does this in three stages:

### Stage 1 — Audio Understanding (notebook 01)

Before any ML, we first understand what audio data looks like as numbers.

**Waveform** — Raw audio is a 1D array of air pressure measurements. At 44,100 samples per second, a 5-second clip contains 220,500 numbers. A dog bark shows up as one sharp spike; a thunderstorm is a long low rumble.

**Mel Spectrogram** — We convert the waveform into a 2D image (128 frequency bands × 216 time frames) using the Mel scale, which mimics human hearing by compressing high frequencies. The result is a visual fingerprint of the sound — each class has a distinctive pattern.

**MFCCs** — Mel-Frequency Cepstral Coefficients compress the spectrogram into 13 numbers per time frame, capturing the essential texture of the sound. Taking the mean across time gives us 13 numbers representing an entire 5-second clip.

---

### Stage 2 — Classical ML Pipeline (notebook 02)

We extract hand-crafted features from each audio clip and train traditional classifiers.

**Features extracted (40 total):**
- 13 MFCC means — average frequency texture
- 13 MFCC standard deviations — how much texture varies over time
- 12 Chroma means — pitch class information
- 1 Spectral centroid — brightness of the sound
- 1 Zero crossing rate — how noisy vs tonal the sound is

**Models trained:**

| Model | Accuracy |
|---|---|
| SVM (13 MFCC means only) | 51.75% |
| SVM (40 rich features) | 64.75% |
| Random Forest | 56.75% |
| XGBoost | 55.00% |

SVM with RBF kernel wins on this dataset because it handles small, well-scaled continuous feature spaces better than tree-based models.

**Best classes:** thunderstorm (0.88), train (0.84), crying baby (0.82)
**Worst classes:** drinking/sipping (0.00), laughing (0.00) — too similar to other classes

---

### Stage 3 — Deep Learning Pipeline (notebook 03)

We convert each audio clip to a Mel Spectrogram and treat it as an image classification problem.

**Dataset class** — A custom PyTorch `Dataset` loads audio files on demand, converts them to Mel Spectrograms, normalizes, and returns tensors of shape `[1, 128, 216]` — one grayscale spectrogram image per clip.

**Custom CNN (EchoNet from scratch)**

```
Input:  [batch, 1, 128, 216]
Conv2d(1→32) + BatchNorm + ReLU + MaxPool  →  [batch, 32, 64, 108]
Conv2d(32→64) + BatchNorm + ReLU + MaxPool →  [batch, 64, 32, 54]
Conv2d(64→128) + BatchNorm + ReLU + MaxPool → [batch, 128, 16, 27]
Flatten → Linear(55296→512) → Dropout(0.5) → Linear(512→50)
```

Achieved 46% after 60 epochs — limited by small dataset size (2,000 clips).

**EfficientNet Transfer Learning**

We load EfficientNet-B0 pretrained on ImageNet and adapt it for audio:
- Replace first Conv2d to accept 1-channel (grayscale) spectrograms instead of 3-channel RGB
- Replace final classifier: `Linear(1280 → 50)`
- Phase 1 (epochs 1–30): freeze backbone, train classifier only → 62%
- Phase 2 (epochs 31–50): unfreeze all layers, fine-tune with lr=0.0001 → 74.5%

The pretrained ImageNet features (edges, textures, patterns) transfer surprisingly well to spectrogram images.

---

## 📊 Training Curves

The training curves tell the full story:

- **EfficientNet loss** drops sharply in the first 5 epochs — pretrained knowledge activates immediately
- **Scratch CNN loss** drops slowly over 60 epochs — learning from random weights is hard with only 2,000 samples
- **EfficientNet accuracy** crosses the SVM baseline around epoch 10 and peaks at 74.5%
- **Scratch CNN** never beats the SVM baseline

---

## 🗃️ Dataset

**ESC-50** — Environmental Sound Classification dataset

- 2,000 audio clips, 5 seconds each, `.wav` format
- 50 classes across 5 categories: animals, natural sounds, human sounds, interior sounds, exterior sounds
- Perfectly balanced — exactly 40 clips per class
- 5 folds for cross-validation

Download: [https://github.com/karoldvl/ESC-50](https://github.com/karoldvl/ESC-50)

---

## ⚙️ Setup

```bash
conda create -n birdclef python=3.10 -y
conda activate birdclef
python -m pip install librosa soundfile numpy pandas scikit-learn matplotlib tqdm joblib
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🚀 Run The App

```bash
cd app
python app.py
```

Upload any `.wav` audio clip and get the top predicted sound category with confidence scores.

---

## 🔮 What's Next

- Data augmentation (time stretch, pitch shift, additive noise) to improve CNN accuracy
- 5-fold cross-validation for more reliable evaluation
- BirdCLEF extension — apply the same pipeline to bird species classification
- Grad-CAM visualization to see which spectrogram regions the CNN focuses on

---

## 🛠️ Tech Stack

Python · PyTorch · librosa · scikit-learn · EfficientNet · Gradio · Streamlit

---

## 👤 Author

**Avik Sarkar** — B.Tech CSE AI/ML, Brainware University (2024–2028)

GitHub: [aviksarkar0204-stack](https://github.com/aviksarkar0204-stack)
Hugging Face: [Avik128](https://huggingface.co/Avik128)
