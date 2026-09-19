import torchaudio

def save_audio(waveform, path, sample_rate):
    torchaudio.save(path, waveform.cpu(), sample_rate)
