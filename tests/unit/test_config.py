"""
Unit tests for the Configuration module.
"""
import pytest
from src.shared.config import Config

def test_config_loads_actual_file():
    """Verifies the Config class correctly parses the setup.json file."""
    config = Config()
    assert config.context_window == 10
    assert len(config.frequencies) == 4
    assert config.sigma_amplitude == 0.1
    assert config.sigma_phase == 0.05
    assert config.version == "1.00"

def test_config_file_not_found():
    """Verifies defensive programming when config is missing."""
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        Config("config/fake_file.json")

def test_config_repr():
    """Verifies the string representation works for logging."""
    config = Config()
    repr_str = repr(config)
    assert "Config(" in repr_str
    assert "window=10" in repr_str
    assert "sigma_amp=0.1" in repr_str