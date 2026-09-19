import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicSynthesizer(nn.Module):
    """
    Differentiable Additive Harmonic Synthesizer.
    Generates audio by summing multiple harmonically related sine waves.
    """
    def __init__(self, sample_rate=16000, harmonics=64, hop_length=160):
        super().__init__()
        self.sample_rate = sample_rate
        self.harmonics = harmonics
        self.hop_length = hop_length
        self.register_buffer("harmonic_numbers", torch.arange(1, harmonics + 1).float())

    def upsample(self, x):
        x = x.transpose(1, 2)
        x = F.interpolate(x, scale_factor=self.hop_length, mode='linear', align_corners=False)
        return x.transpose(1, 2)

    def forward(self, pitch, harmonic_amplitudes):
        pitch_samples = self.upsample(pitch)
        amplitude_samples = self.upsample(harmonic_amplitudes)

        omega = 2 * torch.pi * pitch_samples / self.sample_rate
        phase = torch.cumsum(omega, dim=1)

        harmonic_phases = phase * self.harmonic_numbers
        sines = torch.sin(harmonic_phases)

        frequencies = pitch_samples * self.harmonic_numbers
        anti_alias_mask = (frequencies < (self.sample_rate / 2)).float()
        
        audio = torch.sum(sines * amplitude_samples * anti_alias_mask, dim=-1)
        return audio

class FilteredNoiseSynthesizer(nn.Module):
    """
    Differentiable Subtractive Noise Synthesizer.
    Applies a time-varying FIR filter to white noise.
    """
    def __init__(self, window_size=256, hop_length=160):
        super().__init__()
        self.window_size = window_size
        self.hop_length = hop_length

    def forward(self, noise_magnitudes):
        """
        noise_magnitudes: (batch, num_frames, noise_bands)
        """
        batch_size, num_frames, noise_bands = noise_magnitudes.shape
        num_samples = num_frames * self.hop_length
        
        # 1. Upsample noise magnitudes to the frequency domain window resolution
        # We assume noise_bands maps linearly to frequency bins up to Nyquist
        # For a simple implementation, we treat noise_magnitudes as the amplitude of the frequency response
        # and interpolate it across time frames and frequency bands to match STFT bins.
        
        # Generate white noise
        white_noise = torch.empty(batch_size, num_samples, device=noise_magnitudes.device).uniform_(-1, 1)
        
        # We use STFT/ISTFT for filtering.
        # Compute STFT of white noise
        noise_stft = torch.stft(
            white_noise, 
            n_fft=self.window_size, 
            hop_length=self.hop_length, 
            return_complex=True,
            center=False
        ) # Shape: (batch, num_bins, num_frames)
        
        num_bins = noise_stft.shape[1]
        
        # Interpolate noise magnitudes to match STFT bins (frequency dimension)
        # noise_magnitudes is (batch, num_frames, noise_bands)
        magnitudes = noise_magnitudes.transpose(1, 2) # (batch, noise_bands, num_frames)
        
        # Interpolate across frequency bins
        # Shape becomes (batch, num_bins, num_frames)
        filter_magnitudes = F.interpolate(magnitudes, size=num_frames, mode='linear', align_corners=False)
        filter_magnitudes = F.interpolate(filter_magnitudes.transpose(1,2), size=num_bins, mode='linear', align_corners=False).transpose(1,2)

        # Apply filter to the white noise STFT
        filtered_stft = noise_stft * filter_magnitudes
        
        # Convert back to audio
        filtered_audio = torch.istft(
            filtered_stft, 
            n_fft=self.window_size, 
            hop_length=self.hop_length,
            length=num_samples,
            center=False
        )
        
        return filtered_audio
