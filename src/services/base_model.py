"""
Base Neural Network module providing shared training and weight initialization logic.
"""
import torch
import torch.nn as nn

class BaseNN(nn.Module):
    """Mixin class for shared neural network operations."""
    
    def initialize_weights(self):
        """Applies Xavier/He initialization to linear and recurrent layers."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
            elif isinstance(param, nn.Parameter) and len(param.shape) > 1:
                nn.init.xavier_uniform_(param.data)

    def calculate_mse_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates Mean Squared Error loss."""
        criterion = nn.MSELoss()
        return criterion(predictions, targets)

    def perform_training_step(self, optimizer: torch.optim.Optimizer, 
                              inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Executes a single forward and backward pass."""
        self.train()
        optimizer.zero_grad()
        
        predictions = self(inputs)
        loss = self.calculate_mse_loss(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()