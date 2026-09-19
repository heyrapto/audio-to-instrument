from .pitch import extract_pitch
from .loudness import extract_loudness

def extract_features(waveform, sample_rate):
    pitch, confidence = extract_pitch(waveform, sample_rate)
    loudness = extract_loudness(waveform)
    
    return {
        "pitch": pitch,
        "confidence": confidence,
        "loudness": loudness,
    }
