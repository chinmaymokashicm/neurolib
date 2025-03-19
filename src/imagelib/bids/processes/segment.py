"""
Unitary processes for segmentation pipelines.
"""
from ..pipeline import get_new_pipeline_derived_filename
from .helper import save_sidecar

from pathlib import Path, PosixPath
from typing import Optional