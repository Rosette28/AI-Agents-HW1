"""
Recurrent Neural Network (RNN) architecture.
Processes the signal sequentially to capture temporal dependencies.
"""
import torch
import torch.nn as nn
from src.services.base_model import BaseNN

class RNNModel(BaseNN):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # batch_first=True expects tensors of shape [Batch, Sequence, Features]
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.initialize_weights()

    def init_hidden_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initializes hidden state to zeros."""
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects shape: [Batch, 10, 6]
        Returns shape: [Batch, 10]
        """
        batch_size = x.size(0)
        h0 = self.init_hidden_state(batch_size, x.device)
        
        out, _ = self.rnn(x, h0)
        
        # Apply linear layer to every time step
        out = self.fc(out)
        
        # Squeeze the final feature dimension to match [Batch, 10] targets
        return out.squeeze(-1)