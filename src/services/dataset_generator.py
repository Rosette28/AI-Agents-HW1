"""
Dataset Generator Service for Denoising Sine Waves.
Compliant with Phase 1 PRDs for MLP, RNN, and LSTM data shapes.
"""
import numpy as np

class SineWaveGenerator:
    def __init__(self, config):
        # Gracefully extract from config, with safe fallbacks
        self.freqs = getattr(config, 'frequencies', [1, 2, 5, 10])
        self.sigma = getattr(config, 'sigma_amplitude', 0.1)
        self.sigma2 = getattr(config, 'sigma_phase', 0.05)
        self.window = getattr(config, 'context_window', 10)
        self.num_samples = getattr(config, 'num_samples', 5000)
        self.amplitude = getattr(config, 'amplitude', 1.0)

    def generate_pure_signal(self, freq: float, phi: np.ndarray) -> np.ndarray:
        """Generates the mathematical pure sine wave."""
        return self.amplitude * np.sin(2 * np.pi * freq * phi)

    def apply_noise(self, freq: float, phi: np.ndarray) -> np.ndarray:
        """Applies both amplitude (sigma) and phase (sigma2) noise."""
        amp_noise = np.random.normal(0, self.sigma, size=len(phi))
        phase_noise = np.random.normal(0, self.sigma2, size=len(phi))
        return (self.amplitude + amp_noise) * np.sin(2 * np.pi * freq * phi + phase_noise)

    def _validate_signal_length(self, signal: np.ndarray):
        """Ensures the signal matches the required context window."""
        if len(signal) != self.window:
            raise ValueError(f"Signal length {len(signal)} does not match window {self.window}")

    def create_context_window(self, freq_idx: int):
        """Generates a 10-sample window formatted for all neural architectures."""
        freq = self.freqs[freq_idx]
        phi = np.linspace(0, 1, self.window)
        
        clean = self.generate_pure_signal(freq, phi)
        noisy = self.apply_noise(freq, phi)
        self._validate_signal_length(clean)

        # 1-hot encode frequency
        c_vec = np.zeros(len(self.freqs))
        c_vec[freq_idx] = 1.0

        # MLP Format: [10 noisy, 4 freq, 1 noise] (Flattened 15-dim)
        mlp_input = np.concatenate([noisy, c_vec, [self.sigma]])

        # RNN/LSTM Format: 10 steps of [1 noisy, 4 freq, 1 noise] (Shape: 10x6)
        rnn_input = np.zeros((self.window, 6))
        for t in range(self.window):
            rnn_input[t] = np.concatenate([[noisy[t]], c_vec, [self.sigma]])

        return mlp_input.astype(np.float32), rnn_input.astype(np.float32), clean.astype(np.float32)

    def build_dataset(self):
        """Assembles the final dataset pairs for training."""
        mlp_inputs, rnn_inputs, targets = [], [], []

        for _ in range(self.num_samples):
            freq_idx = np.random.randint(len(self.freqs))
            mlp_x, rnn_x, target_y = self.create_context_window(freq_idx)
            
            mlp_inputs.append(mlp_x)
            rnn_inputs.append(rnn_x)
            targets.append(target_y)

        return np.array(mlp_inputs), np.array(rnn_inputs), np.array(targets)