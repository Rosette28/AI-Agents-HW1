"""
Multi-Layer Perceptron (MLP) architecture for signal denoising.
Processes the entire 10-sample window as a flattened vector.
"""
import torch
import torch.nn as nn
from src.services.base_model import BaseNN

class MLPModel(BaseNN):
    def __init__(self, input_dim: int = 15, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self._build_layers(input_dim, hidden_dim, output_dim)
        self.initialize_weights()

    def _build_layers(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Constructs the fully connected layers dynamically."""
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects shape: [Batch, 15] (10 noisy + 4 freq + 1 noise)
        Returns shape: [Batch, 10]
        """
        return self.network(x)