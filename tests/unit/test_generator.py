"""
Unit tests for the Dataset Generator.
"""
import pytest
import numpy as np
from src.services.dataset_generator import SineWaveGenerator

class MockConfig:
    """Mock configuration to ensure tests run deterministically."""
    frequencies = [1, 2, 5, 10]
    sigma_amplitude = 0.1
    sigma_phase = 0.05
    context_window = 10
    num_samples = 5
    amplitude = 1.0

@pytest.fixture
def generator():
    return SineWaveGenerator(MockConfig())

def test_generate_pure_signal(generator):
    phi = np.linspace(0, 1, 10)
    clean = generator.generate_pure_signal(1.0, phi)
    
    # Verify length and math (sin(0) should be 0)
    assert len(clean) == 10
    assert np.isclose(clean[0], 0.0, atol=1e-5)

def test_create_context_window_dimensions(generator):
    # Pass index 0 (which is frequency 1)
    mlp_in, rnn_in, clean = generator.create_context_window(0)
    
    # MLP must be 1D tensor of 15 (10 signal + 4 one-hot + 1 noise)
    assert mlp_in.shape == (15,)
    
    # RNN must be 2D sequence tensor of 10 steps, each with 6 features
    assert rnn_in.shape == (10, 6)
    
    # Target output must be exactly 10 samples
    assert clean.shape == (10,)

def test_build_dataset_batching(generator):
    mlp_x, rnn_x, target_y = generator.build_dataset()
    
    # Verify the batch size matches num_samples (5)
    assert mlp_x.shape == (5, 15)
    assert rnn_x.shape == (5, 10, 6)
    assert target_y.shape == (5, 10)