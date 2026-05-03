"""
SineDenoisingSDK: The single entry point for the application.
"""
import torch
from torch import optim
from src.shared.config import Config
from src.services.dataset_generator import SineWaveGenerator
from src.services.models_mlp import MLPModel
from src.services.models_rnn import RNNModel
from src.services.models_lstm import LSTMModel

class SineDenoisingSDK:
    def __init__(self):
        self.config = Config()
        self.generator = SineWaveGenerator(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.mlp = MLPModel().to(self.device)
        self.rnn = RNNModel().to(self.device)
        self.lstm = LSTMModel().to(self.device)

    def initialize_dataset(self):
        """Generates the training dataset."""
        print("Generating dataset...")
        mlp_x, rnn_x, targets = self.generator.build_dataset()
        
        # Convert to PyTorch tensors
        self.data = {
            "mlp_x": torch.tensor(mlp_x, dtype=torch.float32).to(self.device),
            "rnn_x": torch.tensor(rnn_x, dtype=torch.float32).to(self.device),
            "targets": torch.tensor(targets, dtype=torch.float32).to(self.device)
        }
        print(f"Dataset generated. Batch shape: {self.data['targets'].shape}")

    def train_models(self, epochs: int = 50):
        """Trains all three models."""
        print(f"\nStarting training for {epochs} epochs on {self.device}...")
        
        # Setup optimizers
        opt_mlp = optim.Adam(self.mlp.parameters(), lr=0.01)
        opt_rnn = optim.Adam(self.rnn.parameters(), lr=0.01)
        opt_lstm = optim.Adam(self.lstm.parameters(), lr=0.01)

        for epoch in range(epochs):
            loss_mlp = self.mlp.perform_training_step(opt_mlp, self.data["mlp_x"], self.data["targets"])
            loss_rnn = self.rnn.perform_training_step(opt_rnn, self.data["rnn_x"], self.data["targets"])
            loss_lstm = self.lstm.perform_training_step(opt_lstm, self.data["rnn_x"], self.data["targets"])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | MLP Loss: {loss_mlp:.4f} | RNN Loss: {loss_rnn:.4f} | LSTM Loss: {loss_lstm:.4f}")
    def execute_denoising(self, mlp_input: torch.Tensor, rnn_input: torch.Tensor) -> dict:
        """Runs inference to denoise new signals using all three models."""
        self.mlp.eval()
        self.rnn.eval()
        self.lstm.eval()
        
        with torch.no_grad():
            return {
                "mlp_out": self.mlp(mlp_input),
                "rnn_out": self.rnn(rnn_input),
                "lstm_out": self.lstm(rnn_input)
            }

    def get_evaluation_report(self) -> dict:
        """Returns the final MSE evaluation metrics for the models."""
        print("Generating evaluation report...")
        results = self.execute_denoising(self.data["mlp_x"], self.data["rnn_x"])
        
        return {
            "MLP_MSE": self.mlp.calculate_mse_loss(results["mlp_out"], self.data["targets"]).item(),
            "RNN_MSE": self.rnn.calculate_mse_loss(results["rnn_out"], self.data["targets"]).item(),
            "LSTM_MSE": self.lstm.calculate_mse_loss(results["lstm_out"], self.data["targets"]).item()
        }