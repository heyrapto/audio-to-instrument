import os
from pathlib import Path

base_dir = Path(".")

# Create directories
dirs = [
    "configs",
    "data/raw/trumpet",
    "data/raw/input",
    "data/processed/trumpet",
    "data/processed/features",
    "data/splits",
    "src/audio",
    "src/features",
    "src/models",
    "src/losses",
    "src/training",
    "src/inference",
    "src/utils",
    "scripts",
    "notebooks",
    "checkpoints",
    "outputs",
    "tests"
]
for d in dirs:
    (base_dir / d).mkdir(parents=True, exist_ok=True)

# Define files and content
files = {
    "README.md": """# Audio to Instrument
A PyTorch-based system for converting audio performances into realistic instrument audio.

## Goal
Input: singing.wav
Output: trumpet.wav

## Architecture
Audio -> preprocessing -> pitch/loudness extraction -> neural expression model -> differentiable trumpet synthesizer -> output audio

## Current status
- [ ] Audio loading
- [ ] Resampling
- [ ] Pitch extraction
- [ ] Loudness extraction
- [ ] Onset detection
- [ ] Trumpet dataset
- [ ] Baseline synthesizer
- [ ] Neural model
- [ ] Training
- [ ] Evaluation
- [ ] Inference

## Installation
...
## Training
...
## Inference
...
## Dataset
...
## License
...
""",
    "requirements.txt": """torch
torchaudio
torchcrepe
numpy
scipy
librosa
soundfile
pyyaml
tqdm
matplotlib
pandas
scikit-learn
pytest
jupyter
""",
    "pyproject.toml": """[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "audio-to-instrument"
version = "0.1.0"
description = "Audio to instrument conversion using PyTorch"
requires-python = ">=3.10"

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
""",
    ".gitignore": """__pycache__/
*.py[cod]
.venv/
venv/
.env
.ipynb_checkpoints/
data/raw/*
data/processed/*
checkpoints/*
outputs/*
*.wav
*.mp3
*.flac
*.pt
*.pth
*.ckpt
.DS_Store
!data/raw/trumpet/.gitkeep
""",
    "data/raw/trumpet/.gitkeep": "",
    "configs/base.yaml": """audio:
  sample_rate: 16000
  channels: 1
  hop_length: 160
  n_fft: 1024
features:
  fmin: 50
  fmax: 1200
  pitch_confidence_threshold: 0.5
model:
  hidden_size: 256
  num_layers: 4
paths:
  raw_data: data/raw
  processed_data: data/processed
  features: data/processed/features
  checkpoints: checkpoints
  outputs: outputs
""",
    "configs/trumpet.yaml": """instrument:
  name: trumpet
  fmin: 80
  fmax: 1200
  harmonic_count: 64
  noise_bands: 8
  output_sample_rate: 16000
""",
    "configs/training.yaml": """training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.0003
  weight_decay: 0.00001
  num_workers: 2
  save_every: 5
  mixed_precision: true
  gradient_clip: 1.0
validation:
  interval: 1
""",
    "data/splits/train.txt": "",
    "data/splits/validation.txt": "",
    "data/splits/test.txt": "",
    "src/audio/loader.py": """import torch
import torchaudio

def load_audio(path: str):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate
""",
    "src/audio/resampling.py": """import torchaudio

def resample_audio(waveform, original_sr: int, target_sr: int):
    if original_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(original_sr, target_sr)
    return resampler(waveform)
""",
    "src/audio/preprocessing.py": """import torch
from .loader import load_audio
from .resampling import resample_audio

def preprocess_audio(path: str, target_sr: int = 16000):
    waveform, sr = load_audio(path)
    
    # Convert stereo -> mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
        
    waveform = resample_audio(waveform, sr, target_sr)
    
    # Normalize
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
        
    return waveform
""",
    "src/audio/augmentation.py": """import torch

def add_noise(waveform: torch.Tensor, amount: float = 0.005):
    noise = torch.randn_like(waveform)
    return waveform + noise * amount

def random_gain(waveform: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2):
    gain = torch.empty(1).uniform_(min_gain, max_gain)
    return waveform * gain
""",
    "src/features/pitch.py": """import torch
import torchcrepe

def extract_pitch(waveform: torch.Tensor, sample_rate: int, fmin: float = 50, fmax: float = 1200):
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
        
    pitch, periodicity = torchcrepe.predict(
        waveform,
        sample_rate,
        hop_length=160,
        fmin=fmin,
        fmax=fmax,
        model="tiny",
        decoder=torchcrepe.decode.weighted_argmax,
        return_periodicity=True,
    )
    return pitch, periodicity
""",
    "src/features/loudness.py": """import torch

def extract_loudness(waveform: torch.Tensor, frame_size: int = 1024, hop_length: int = 160):
    waveform = waveform.squeeze(0)
    frames = waveform.unfold(0, frame_size, hop_length)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
    loudness = 20 * torch.log10(rms)
    return loudness
""",
    "src/features/onset.py": """import librosa
import numpy as np

def detect_onsets(waveform: np.ndarray, sample_rate: int):
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sample_rate)
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    return onset_times
""",
    "src/features/extraction.py": """from .pitch import extract_pitch
from .loudness import extract_loudness

def extract_features(waveform, sample_rate):
    pitch, confidence = extract_pitch(waveform, sample_rate)
    loudness = extract_loudness(waveform)
    
    return {
        "pitch": pitch,
        "confidence": confidence,
        "loudness": loudness,
    }
""",
    "src/models/pitch_encoder.py": """import torch
import torch.nn as nn

class PitchEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, pitch):
        return self.network(pitch)
""",
    "src/models/expression_encoder.py": """import torch.nn as nn

class ExpressionEncoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.network(x)
""",
    "src/models/encoder.py": """import torch
import torch.nn as nn
from .pitch_encoder import PitchEncoder
from .expression_encoder import ExpressionEncoder

class AudioEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.pitch = PitchEncoder(hidden_size)
        self.expression = ExpressionEncoder(hidden_size=hidden_size)
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, pitch, loudness):
        pitch_features = self.pitch(pitch)
        expression_features = self.expression(loudness)
        
        x = torch.cat([pitch_features, expression_features], dim=-1)
        return self.combine(x)
""",
    "src/models/synthesizer.py": """import torch
import torch.nn as nn

class HarmonicSynthesizer(nn.Module):
    def __init__(self, sample_rate=16000, harmonics=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonics = harmonics
        self.amplitudes = nn.Parameter(torch.randn(harmonics))

    def forward(self, pitch, amplitude):
        # Implementation will generate
        # harmonic partials from F0.
        pass
""",
    "src/models/trumpet_model.py": """import torch
import torch.nn as nn
from .encoder import AudioEncoder
from .synthesizer import HarmonicSynthesizer

class TrumpetModel(nn.Module):
    def __init__(self, hidden_size=256, sample_rate=16000):
        super().__init__()
        self.encoder = AudioEncoder(hidden_size)
        self.controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128),
        )
        self.synthesizer = HarmonicSynthesizer(sample_rate=sample_rate)

    def forward(self, pitch, loudness):
        latent = self.encoder(pitch, loudness)
        controls = self.controller(latent)
        audio = self.synthesizer(pitch, controls)
        return audio
""",
    "src/losses/waveform.py": """import torch.nn.functional as F

def waveform_loss(predicted, target):
    return F.l1_loss(predicted, target)
""",
    "src/losses/spectral.py": """import torch
import torch.nn.functional as F

def spectral_loss(predicted, target, n_fft=1024, hop_length=256):
    predicted_spec = torch.stft(
        predicted,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    ).abs()
    
    target_spec = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        return_complex=True,
    ).abs()
    
    return F.l1_loss(predicted_spec, target_spec)
""",
    "src/losses/total.py": """from .waveform import waveform_loss
from .spectral import spectral_loss

def total_loss(predicted, target):
    waveform = waveform_loss(predicted, target)
    spectral = spectral_loss(predicted, target)
    return 0.5 * waveform + 1.0 * spectral
""",
    "src/training/dataset.py": """import torch
from torch.utils.data import Dataset

class InstrumentDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "audio": torch.tensor(sample["audio"], dtype=torch.float32),
            "pitch": torch.tensor(sample["pitch"], dtype=torch.float32),
            "loudness": torch.tensor(sample["loudness"], dtype=torch.float32),
        }
""",
    "src/training/trainer.py": """class Trainer:
    def __init__(self, model, optimizer, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        predicted = self.model(
            batch["pitch"].to(self.device),
            batch["loudness"].to(self.device),
        )
        
        loss = self.loss_fn(
            predicted,
            batch["audio"].to(self.device),
        )
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
""",
    "src/training/checkpoint.py": """import torch

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": loss,
        },
        path,
    )

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]
""",
    "src/training/metrics.py": """def mean(values):
    return sum(values) / len(values)
""",
    "src/inference/pipeline.py": """import torch
from src.audio.preprocessing import preprocess_audio
from src.features.extraction import extract_features
from src.utils.audio import save_audio

def convert(input_path, output_path, model, config):
    waveform = preprocess_audio(input_path, config.sample_rate)
    features = extract_features(waveform, config.sample_rate)
    
    with torch.no_grad():
        output = model(
            features["pitch"],
            features["loudness"],
        )
        
    save_audio(output, output_path, config.sample_rate)
""",
    "src/inference/generate.py": """import torch
import torchaudio

def generate_audio(model, pitch, loudness, output_path, sample_rate):
    model.eval()
    with torch.no_grad():
        audio = model(pitch, loudness)
        
    torchaudio.save(output_path, audio.cpu(), sample_rate)
""",
    "src/utils/config.py": """import yaml

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)
""",
    "src/utils/audio.py": """import torchaudio

def save_audio(waveform, path, sample_rate):
    torchaudio.save(path, waveform.cpu(), sample_rate)
""",
    "src/utils/seed.py": """import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
""",
    "src/utils/logging.py": """import logging

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)
    return logger
""",
    "scripts/download_dataset.py": """def download():
    # download dataset
    # verify archive
    # extract
    # organize files
    pass
""",
    "scripts/prepare_dataset.py": """from pathlib import Path
from src.audio.preprocessing import preprocess_audio
from src.utils.audio import save_audio

def main():
    raw_dir = Path("data/raw/trumpet")
    output_dir = Path("data/processed/trumpet")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for path in raw_dir.glob("*.wav"):
        waveform = preprocess_audio(str(path))
        save_audio(waveform, output_dir / path.name, 16000)

if __name__ == "__main__":
    main()
""",
    "scripts/extract_features.py": """import numpy as np

def extract_and_save():
    # Loop over processed audio files, extract features and save to npz
    pass
""",
    "scripts/train.py": """import torch
from src.models.trumpet_model import TrumpetModel
from src.losses.total import total_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrumpetModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for epoch in range(100):
        # load batches
        # forward
        # calculate loss
        # backward
        # optimizer step
        print(f"Epoch {epoch}")

if __name__ == "__main__":
    main()
""",
    "scripts/evaluate.py": """def evaluate():
    # Load model and evaluate against target samples
    pass
""",
    "scripts/convert.py": """import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    # load model
    # run inference
    # save audio

if __name__ == "__main__":
    main()
""",
    "tests/test_audio.py": """def test_audio_loading():
    pass
""",
    "tests/test_features.py": """def test_pitch_extraction():
    pass
""",
    "tests/test_model.py": """from src.models.trumpet_model import TrumpetModel

def test_model_forward():
    model = TrumpetModel()
    # pitch = ...
    # loudness = ...
    # output = model(pitch, loudness)
    # assert output is not None
""",
    "tests/test_pipeline.py": """def test_pipeline():
    pass
""",
}

for filepath, content in files.items():
    with open(base_dir / filepath, 'w') as f:
        f.write(content)
        
print("Project structure created successfully.")
