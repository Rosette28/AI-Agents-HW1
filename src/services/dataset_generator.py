import numpy as np
from src.shared.config import Config


class SineWaveDatasetGenerator:
    """
    Generates sine wave dataset with noise for training models.
    Each sample includes:
    - Input: [10 noisy values + one-hot frequency (4) + noise level (1)]
    - Target: 10 clean values
    """

    def __init__(self, config: Config):
        self.config = config
        self.freqs = config.frequencies
        self.noise_levels = config.noise_levels
        self.window = config.context_window
        self.amplitude = config.amplitude

    def _one_hot(self, index: int, size: int):
        vec = np.zeros(size)
        vec[index] = 1.0
        return vec

    def _generate_signal(self, freq: float, noise: float):
        phi = np.linspace(0, 1, self.window)

        clean = self.amplitude * np.sin(2 * np.pi * freq * phi)

        noise_vec = np.random.normal(0, noise, size=self.window)
        noisy = clean + noise_vec

        return noisy, clean

    def generate_sample(self):
        # Choose random frequency and noise
        freq_idx = np.random.randint(len(self.freqs))
        freq = self.freqs[freq_idx]
        noise = np.random.choice(self.noise_levels)

        # Generate signals
        noisy, clean = self._generate_signal(freq, noise)

        # Build input vector
        freq_one_hot = self._one_hot(freq_idx, len(self.freqs))

        input_vec = np.concatenate([noisy, freq_one_hot, [noise]])

        return input_vec.astype(np.float32), clean.astype(np.float32)

    def generate_dataset(self):
        inputs = []
        targets = []

        for _ in range(self.config.num_samples):
            x, y = self.generate_sample()
            inputs.append(x)
            targets.append(y)

        return np.array(inputs), np.array(targets)