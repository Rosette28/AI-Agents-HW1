"""
Unit tests for Neural Networks and SDK.
"""
import torch
from src.services.models_mlp import MLPModel
from src.services.models_rnn import RNNModel
from src.services.models_lstm import LSTMModel
from src.sdk.sdk import SineDenoisingSDK

def test_mlp_tensor_dimensions():
    """Test 3.2.5: Input shape [Batch, 15] -> Output [Batch, 10]"""
    model = MLPModel()
    mock_input = torch.randn(32, 15)
    output = model(mock_input)
    assert output.shape == (32, 10)

def test_sequence_models_dimensions():
    """Test 3.3.8: RNN/LSTM sequence processing."""
    rnn = RNNModel()
    lstm = LSTMModel()
    mock_input = torch.randn(32, 10, 6) # [Batch, Seq=10, Features=6]
    
    assert rnn(mock_input).shape == (32, 10)
    assert lstm(mock_input).shape == (32, 10)

def test_sdk_end_to_end_pipeline():
    """Test 3.4.8: Complete pipeline execution."""
    sdk = SineDenoisingSDK()
    # Shrink dataset size just for the test to make it run instantly
    sdk.config.num_samples = 32 
    
    sdk.initialize_dataset()
    sdk.train_models(epochs=1)
    
    report = sdk.get_evaluation_report()
    assert "MLP_MSE" in report
    assert "RNN_MSE" in report
    assert "LSTM_MSE" in report