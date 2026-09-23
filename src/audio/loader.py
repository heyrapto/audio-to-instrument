import torch
import soundfile as sf

def load_audio(path: str):
    data, sample_rate = sf.read(path)
    # soundfile returns (frames, channels) or just (frames,)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    # torch expects (channels, frames)
    waveform = torch.tensor(data, dtype=torch.float32).t()
    return waveform, sample_rate
