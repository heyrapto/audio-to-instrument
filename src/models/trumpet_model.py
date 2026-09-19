import torch
import torch.nn as nn
from .encoder import AudioEncoder
from .synthesizer import HarmonicSynthesizer, FilteredNoiseSynthesizer

class TrumpetModel(nn.Module):
    """
    End-to-end Trumpet model.
    Maps extracted audio features (pitch, loudness) -> latent controls -> synthesizer -> audio.
    """
    def __init__(self, hidden_size=256, sample_rate=16000, harmonics=64, noise_bands=65, hop_length=160):
        super().__init__()
        self.encoder = AudioEncoder(hidden_size)
        
        self.controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        self.proj_amplitudes = nn.Linear(256, harmonics)
        self.proj_global_amp = nn.Linear(256, 1)
        self.proj_noise = nn.Linear(256, noise_bands)
        
        self.harmonic_synth = HarmonicSynthesizer(
            sample_rate=sample_rate, 
            harmonics=harmonics, 
            hop_length=hop_length
        )
        
        # Noise window size is usually (noise_bands - 1) * 2
        window_size = (noise_bands - 1) * 2 if noise_bands > 1 else 256
        self.noise_synth = FilteredNoiseSynthesizer(
            window_size=window_size,
            hop_length=hop_length
        )

    def forward(self, pitch, loudness):
        """
        pitch: (batch, num_frames) or (batch, num_frames, 1)
        loudness: (batch, num_frames) or (batch, num_frames, 1)
        """
        if pitch.dim() == 2:
            pitch = pitch.unsqueeze(-1)
        if loudness.dim() == 2:
            loudness = loudness.unsqueeze(-1)
            
        latent = self.encoder(pitch, loudness)
        controls = self.controller(latent)
        
        # Harmonic Controls
        harmonic_distribution = torch.softmax(self.proj_amplitudes(controls), dim=-1)
        global_amp = torch.sigmoid(self.proj_global_amp(controls))
        harmonic_amplitudes = harmonic_distribution * global_amp
        
        # Noise Controls (positive magnitudes)
        noise_magnitudes = torch.sigmoid(self.proj_noise(controls))
        
        # Synthesize audio
        harmonic_audio = self.harmonic_synth(pitch, harmonic_amplitudes)
        noise_audio = self.noise_synth(noise_magnitudes)
        
        # Final audio output
        # Handle potential slight length mismatches due to ISTFT
        min_len = min(harmonic_audio.shape[-1], noise_audio.shape[-1])
        return harmonic_audio[..., :min_len] + noise_audio[..., :min_len]
