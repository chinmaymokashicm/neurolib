"""
Helper functions for BIDS processes.
"""

from pathlib import Path, PosixPath
import json

def save_sidecar(sidecar: dict, sidecar_filepath: str) -> None:
    """
    Save sidecar to a JSON file.
    """
    Path(sidecar_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_filepath, "w") as f:
        json.dump(sidecar, f, indent=4)