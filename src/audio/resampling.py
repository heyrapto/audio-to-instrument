import torchaudio

def resample_audio(waveform, original_sr: int, target_sr: int):
    if original_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(original_sr, target_sr)
    return resampler(waveform)
