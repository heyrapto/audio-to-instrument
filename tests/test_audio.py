import torch
from src.audio.augmentation import add_noise, random_gain

def test_audio_augmentation():
    waveform = torch.zeros(1, 16000)
    
    noisy = add_noise(waveform, amount=0.1)
    assert noisy.shape == waveform.shape
    assert not torch.allclose(noisy, waveform)
    
    gain = random_gain(torch.ones(1, 16000), min_gain=0.5, max_gain=0.9)
    assert gain.shape == waveform.shape
    assert gain.max().item() <= 0.9
    assert gain.min().item() >= 0.5
