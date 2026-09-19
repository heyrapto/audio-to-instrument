from .waveform import waveform_loss
from .spectral import spectral_loss

def total_loss(predicted, target):
    waveform = waveform_loss(predicted, target)
    spectral = spectral_loss(predicted, target)
    return 0.5 * waveform + 1.0 * spectral
