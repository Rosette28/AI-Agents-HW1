import json
from pathlib import Path


class Config:
    """
    Configuration loader for the project.

    Loads parameters from config/setup.json and exposes them
    as attributes for easy access across the project.
    """

    def __init__(self, config_path: str = "config/setup.json"):
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.frequencies = data["frequencies"]
        self.noise_levels = data["noise_levels"]
        self.amplitude = data["amplitude"]
        self.context_window = data["context_window"]
        self.num_samples = data["num_samples"]
        self.batch_size = data["batch_size"]

    def __repr__(self):
        return (
            f"Config(freq={self.frequencies}, noise={self.noise_levels}, "
            f"window={self.context_window}, samples={self.num_samples})"
        )