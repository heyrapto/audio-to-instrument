import torch
import torch.nn as nn

class PitchEncoder(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, pitch):
        return self.network(pitch)
