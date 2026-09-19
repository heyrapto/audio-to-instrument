from src.models.trumpet_model import TrumpetModel
import torch

def test_model_forward():
    model = TrumpetModel()
    
    batch_size = 2
    num_frames = 100
    hop_length = 160
    
    pitch = torch.ones(batch_size, num_frames, 1) * 440.0
    loudness = torch.ones(batch_size, num_frames, 1) * -20.0
    
    output = model(pitch, loudness)
    
    # Check output shape: (batch_size, num_frames * hop_length)
    assert output is not None
    assert output.shape == (batch_size, num_frames * hop_length)
