import torch
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
