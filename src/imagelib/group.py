"""
Work with multiple patients data
"""
from .atlas import AtlasParcellation

from pathlib import Path, PosixPath
from typing import Optional

from bids import BIDSLayout
import pandas as pd
from rich.progress import track

def get_matching_derivative_files(layout: BIDSLayout, scan_filters: dict, derivative_filters: dict) -> dict:
    """
    Get matching derivative files for each scan file.
    
    Args:
        layout: BIDSLayout object containing the BIDS dataset.
        scan_filters: Dictionary containing filters to apply to the BIDSLayout.
        derivative_filters: Dictionary containing filters to apply to the derivatives.
    
    Returns:
        A dictionary containing matching and non-matching scan-derivative pairs.
    """
    scan_derivative_pairs: dict = {"matches": {}, "non_matches": []}
    scan_filters["return_type"] = "file"
    scan_filters["extension"] = "nii.gz"
    scan_files: list[PosixPath] = layout.get(**scan_filters)
    for scan_file in scan_files:
        scan_filters = layout.parse_file_entities(scan_file)
        scan_filters["return_type"] = "file"
        scan_filters["extension"] = "nii.gz"
        # Add derivative filters to the extracted entities and overwrite the existing values
        scan_filters.update(derivative_filters)
        derivative_files = layout.get(**scan_filters)
        derivative_file = derivative_files[0] if derivative_files else None
        if derivative_file:
            scan_derivative_pairs["matches"][scan_file] = derivative_file
        else:
            scan_derivative_pairs["non_matches"].append(scan_file)
    return scan_derivative_pairs


def compute_region_metrics_patients(layout: BIDSLayout, atlas_path: str | PosixPath, scan_filters: dict, mask_filters: Optional[dict] = None, with_masks_only: bool = False) -> list[dict]:
    """
    Compute region-wise metrics for multiple patients.
    
    Args:
        layout: BIDSLayout object containing the BIDS dataset.
        atlas_path: Path to the brain atlas file.
        scan_filters: Dictionary containing filters to apply to the BIDSLayout.
        mask_filters: Dictionary containing filters to apply to the masks. Defaults to None.
        with_masks_only: Whether to compute metrics only for scans with masks. Defaults to False.
        
    Returns:
        A list of dictionaries containing region-wise metrics for each patient.
    """
    scan_derivative_matches: dict = get_matching_derivative_files(layout, scan_filters, mask_filters)
    file_pairs: list[tuple[PosixPath, PosixPath]] = list(scan_derivative_matches["matches"].items())
    if not with_masks_only:
        file_pairs += [(scan_file, None) for scan_file in scan_derivative_matches["non_matches"]]
    atlas_path: PosixPath = Path(atlas_path)
    all_metrics: list[dict] = []
    for scan_file, lesion_mask_file in track(file_pairs, description="Computing region-wise metrics"):
        atlas = AtlasParcellation(atlas_path=atlas_path, brain_scan_path=scan_file, mask_path=lesion_mask_file)
        subject: str = layout.parse_file_entities(scan_file)["subject"]
        session: str = layout.parse_file_entities(scan_file)["session"]
        metrics: list[dict] = atlas.compute_region_metrics()
        for metric_idx in range(len(metrics)):
            metrics[metric_idx]["subject"] = subject
            metrics[metric_idx]["session"] = session
            metrics[metric_idx]["scan_file"] = scan_file
            metrics[metric_idx]["lesion_mask_file"] = lesion_mask_file
        all_metrics += metrics
    return all_metrics
        