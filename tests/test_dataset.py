import numpy as np
from src.shared.config import Config
from src.services.dataset_generator import SineWaveGenerator


def test_dataset_shapes():
    config = Config()
    generator = SineWaveGenerator(config)

    mlp_X, rnn_X, Y = generator.build_dataset()

    assert mlp_X.shape[1] == 15
    assert rnn_X.shape[1:] == (10, 6)
    assert Y.shape[1] == 10


def test_dataset_size():
    config = Config()
    generator = SineWaveGenerator(config)

    mlp_X, rnn_X, Y = generator.build_dataset()

    assert mlp_X.shape[0] == config.num_samples
    assert rnn_X.shape[0] == config.num_samples
    assert Y.shape[0] == config.num_samples


def test_zero_noise():
    config = Config()
    config.sigma_amplitude = 0.0
    config.sigma_phase = 0.0

    generator = SineWaveGenerator(config)
    mlp_X, _, Y = generator.build_dataset()

    noisy_part = mlp_X[:, :10]

    assert np.allclose(noisy_part, Y, atol=1e-5)


def test_one_hot_encoding():
    config = Config()
    generator = SineWaveGenerator(config)

    mlp_X, _, _ = generator.build_dataset()

    one_hot_part = mlp_X[:, 10:14]

    for row in one_hot_part:
        assert np.sum(row) == 1.0