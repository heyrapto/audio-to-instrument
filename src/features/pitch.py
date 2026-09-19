import torch
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
