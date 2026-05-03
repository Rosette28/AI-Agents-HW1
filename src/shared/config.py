import json
from pathlib import Path

class Config:
    """
    Configuration loader for the project.
    Loads parameters from config/setup.json.
    """
    def __init__(self, config_path: str = "config/setup.json"):
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Updated to match the PRD mathematically
        self.version = data.get("version", "1.00")
        self.frequencies = data["frequencies"]
        self.sigma_amplitude = data["sigma_amplitude"]
        self.sigma_phase = data["sigma_phase"]
        self.amplitude = data["amplitude"]
        self.context_window = data["context_window"]
        self.num_samples = data["num_samples"]
        self.batch_size = data["batch_size"]
        self.requests_per_minute = data.get("requests_per_minute", 60)

    def __repr__(self):
        return (
            f"Config(version={self.version}, freq={self.frequencies}, "
            f"sigma_amp={self.sigma_amplitude}, sigma_phase={self.sigma_phase}, "
            f"window={self.context_window}, samples={self.num_samples})"
        )