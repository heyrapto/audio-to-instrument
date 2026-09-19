import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class InstrumentDataset(Dataset):
    """
    Loads pre-extracted features (.npz) for fast training.
    """
    def __init__(self, data_dir: str, segment_frames: int = 250, hop_length: int = 160):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.files = list(self.data_dir.glob("*.npz"))
        self.segment_frames = segment_frames
        self.hop_length = hop_length
        self.segment_samples = segment_frames * hop_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        data = np.load(file_path)
        
        audio = data["audio"]
        pitch = data["pitch"]
        loudness = data["loudness"]
        
        # Ensure dimensions match
        # We need to slice a random segment to ensure consistent batch sizing
        total_frames = pitch.shape[-1]
        
        if total_frames > self.segment_frames:
            # Random crop
            start_frame = np.random.randint(0, total_frames - self.segment_frames)
            start_sample = start_frame * self.hop_length
            
            pitch = pitch[..., start_frame : start_frame + self.segment_frames]
            loudness = loudness[..., start_frame : start_frame + self.segment_frames]
            audio = audio[..., start_sample : start_sample + self.segment_samples]
        else:
            # Pad if too short (rare but possible)
            pad_frames = self.segment_frames - total_frames
            pad_samples = self.segment_samples - audio.shape[-1]
            
            pitch = np.pad(pitch, (0, pad_frames), mode='edge')
            loudness = np.pad(loudness, (0, pad_frames), constant_values=-100.0)
            audio = np.pad(audio, (0, pad_samples))

        return {
            "audio": torch.tensor(audio, dtype=torch.float32),
            "pitch": torch.tensor(pitch, dtype=torch.float32),
            "loudness": torch.tensor(loudness, dtype=torch.float32),
        }
