import torch
import torch.nn.functional as F

class MultiScaleSpectralLoss(torch.nn.Module):
    """
    Multi-resolution STFT loss, standard in state-of-the-art DDSP and vocoders.
    Calculates L1 distance of magnitudes and log-magnitudes across multiple FFT sizes.
    """
    def __init__(self, fft_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = [size // 4 for size in fft_sizes]
        self.win_lengths = fft_sizes

    def forward(self, predicted, target):
        loss = 0.0
        
        # Ensure 2D (batch, time)
        if predicted.dim() == 3:
            predicted = predicted.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            # Compute STFT
            pred_stft = torch.stft(
                predicted, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                return_complex=True, center=True
            ).abs()
            
            target_stft = torch.stft(
                target, n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                return_complex=True, center=True
            ).abs()
            
            # Linear Magnitude Loss (Spectral Convergence)
            linear_loss = F.l1_loss(pred_stft, target_stft)
            
            # Log Magnitude Loss
            log_pred = torch.log(pred_stft + 1e-7)
            log_target = torch.log(target_stft + 1e-7)
            log_loss = F.l1_loss(log_pred, log_target)
            
            loss += linear_loss + log_loss
            
        return loss / len(self.fft_sizes)

def spectral_loss(predicted, target):
    loss_fn = MultiScaleSpectralLoss()
    return loss_fn(predicted, target)
