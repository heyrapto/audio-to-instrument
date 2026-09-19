import librosa
import numpy as np

def detect_onsets(waveform: np.ndarray, sample_rate: int):
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sample_rate)
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate)
    return onset_times
