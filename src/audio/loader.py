import torch
import torchaudio

def load_audio(path: str):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate
