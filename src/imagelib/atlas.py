"""
Work with custom brain atlases and compute parcel/region-wise metrics.
"""
from pathlib import Path, PosixPath
from typing import Optional, Tuple, Dict, Callable
import warnings

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
import nibabel as nib
from nilearn.image import resample_to_img

class AtlasParcellation(BaseModel):
    atlas_path: PosixPath = Field(..., description="Path to the brain atlas file")
    brain_scan_path: PosixPath = Field(..., description="Path to the brain scan file")
    mask_path: Optional[PosixPath] = Field(None, description="Path to the lesion mask file")
    label_map_path: Optional[PosixPath] = Field(None, description="Path to the label map file")
    
    def load_data(self) -> Tuple[np.array, np.array, np.array]:
        atlas = nib.load(self.atlas_path)
        brain = nib.load(self.brain_scan_path)
        mask = nib.load(self.mask_path) if self.mask_path is not None else None
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            
            # Resample mask and scan to match atlas
            resampled_brain = resample_to_img(brain, atlas, force_resample=True, copy_header=True, interpolation="nearest")
            resampled_mask = None
            if mask is not None:
                resampled_mask = resample_to_img(mask, atlas, force_resample=True, copy_header=True, interpolation="nearest")
        
        atlas_data = atlas.get_fdata()
        brain_data = resampled_brain.get_fdata()
        mask_data = None
        if resampled_mask is not None:
            mask_data = resampled_mask.get_fdata()
            mask_data = (mask_data > 0).astype(int)
        
        return atlas_data, brain_data, mask_data
    
    def load_labels(self) -> Dict[int, str]:
        """
        Load region labels from a label map file.
        
        Returns:
            A dictionary mapping region index to anatomical name.
        """
        if not self.label_map_path:
            return {}

        label_dict = {}
        with open(self.label_map_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    index, name = parts
                    label_dict[int(index)] = name.strip()

        return label_dict
    
    def compute_region_metrics(self) -> list[dict]:
        """
        Compute region-wise metrics based on the atlas, brain scan, and optional lesion mask.

        Returns:
            A list of dictionaries containing region-wise metrics.
        """
        atlas_data, brain_data, mask_data = self.load_data()
        region_labels: dict = self.load_labels()

        # Extract unique regions (ignoring background labeled as 0)
        unique_regions = np.unique(atlas_data)
        unique_regions = unique_regions[unique_regions > 0]

        metrics = []

        for region in unique_regions:
            # Create mask for the current region
            region_mask = atlas_data == region

            # Compute metrics for the region
            region_volume = np.sum(region_mask)  # Total number of voxels in the region

            lesion_volume = 0
            lesion_burden = 0
            mean_intensity = brain_data[region_mask].mean()

            # If mask data exists, calculate lesion-related metrics
            if mask_data is not None:
                lesion_region = region_mask & (mask_data == 1)
                lesion_volume = np.sum(lesion_region)
                lesion_burden = lesion_volume / region_volume if region_volume > 0 else 0

            # Store metrics
            region_name = region_labels.get(int(region), f"Region {int(region)}")
            metrics.append(
                {
                    "Region Name": region_name,
                    "Region Volume (voxels)": region_volume,
                    "Lesion Volume (voxels)": lesion_volume,
                    "Lesion Burden": lesion_burden,
                    "Mean Intensity": mean_intensity,
                }
            )

        return metrics