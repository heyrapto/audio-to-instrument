import torch
from src.features.loudness import extract_loudness

def test_loudness_extraction():
    # 1 second of silence
    silence = torch.zeros(1, 16000)
    loudness_silence = extract_loudness(silence)
    
    # 1 second of noise
    noise = torch.randn(1, 16000)
    loudness_noise = extract_loudness(noise)
    
    # Noise should be significantly louder than silence
    assert loudness_noise.mean().item() > loudness_silence.mean().item()
    
    # Check expected shapes (16000 samples, hop_length 160)
    # Number of frames is roughly 16000 // 160
    assert loudness_silence.dim() == 1
    assert abs(loudness_silence.shape[0] - (16000 // 160)) <= 10
