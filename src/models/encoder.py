import torch
import torch.nn as nn
from .pitch_encoder import PitchEncoder
from .expression_encoder import ExpressionEncoder

class AudioEncoder(nn.Module):
    def __init__(self, hidden_size=256):
        super().__init__()
        self.pitch = PitchEncoder(hidden_size)
        self.expression = ExpressionEncoder(hidden_size=hidden_size)
        
        self.combine = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, pitch, loudness):
        pitch_features = self.pitch(pitch)
        expression_features = self.expression(loudness)
        
        x = torch.cat([pitch_features, expression_features], dim=-1)
        return self.combine(x)
