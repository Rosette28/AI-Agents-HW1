"""
Long Short-Term Memory (LSTM) architecture.
Utilizes gating mechanisms to isolate noise from pure frequencies over time.
"""
import torch
import torch.nn as nn
from src.services.base_model import BaseNN

class LSTMModel(BaseNN):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.initialize_weights()

    def init_hidden_state(self, batch_size: int, device: torch.device) -> tuple:
        """Initializes both hidden state and cell state to zeros."""
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects shape: [Batch, 10, 6]
        Returns shape: [Batch, 10]
        """
        batch_size = x.size(0)
        h0, c0 = self.init_hidden_state(batch_size, x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        
        return out.squeeze(-1)