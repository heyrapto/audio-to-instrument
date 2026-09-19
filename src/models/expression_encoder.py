import torch.nn as nn

class ExpressionEncoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.network(x)
