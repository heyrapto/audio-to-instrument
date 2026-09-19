import torch
from src.models.trumpet_model import TrumpetModel
from src.losses.spectral import MultiScaleSpectralLoss

def test_pipeline_forward_and_loss():
    # Simulate a full forward pass and loss computation
    batch_size = 2
    num_frames = 100
    hop_length = 160
    
    # 1. Dummy features
    pitch = torch.ones(batch_size, num_frames, 1) * 440.0
    loudness = torch.ones(batch_size, num_frames, 1) * -20.0
    
    # 2. Model forward
    model = TrumpetModel(hop_length=hop_length)
    output_audio = model(pitch, loudness)
    
    # Verify shape
    assert output_audio.shape == (batch_size, num_frames * hop_length)
    
    # 3. Loss computation
    target_audio = torch.randn(batch_size, num_frames * hop_length)
    loss_fn = MultiScaleSpectralLoss()
    
    loss = loss_fn(output_audio, target_audio)
    
    # Verify loss is a valid scalar and allows backprop
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    
    loss.backward()
    
    # Verify gradients flowed to the controller
    assert model.controller[0].weight.grad is not None
