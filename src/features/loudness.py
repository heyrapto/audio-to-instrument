import torch

def extract_loudness(waveform: torch.Tensor, frame_size: int = 1024, hop_length: int = 160):
    waveform = waveform.squeeze(0)
    frames = waveform.unfold(0, frame_size, hop_length)
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)
    loudness = 20 * torch.log10(rms)
    return loudness
