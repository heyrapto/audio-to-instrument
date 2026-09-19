import torch

def add_noise(waveform: torch.Tensor, amount: float = 0.005):
    noise = torch.randn_like(waveform)
    return waveform + noise * amount

def random_gain(waveform: torch.Tensor, min_gain: float = 0.8, max_gain: float = 1.2):
    gain = torch.empty(1).uniform_(min_gain, max_gain)
    return waveform * gain
