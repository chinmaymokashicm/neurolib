"""
Helper functions for BIDS processes.
"""

from pathlib import Path, PosixPath
import json
import ants
import nibabel as nib

import numpy as np

def save_sidecar(sidecar: dict, sidecar_filepath: str) -> None:
    """
    Save sidecar to a JSON file.
    """
    Path(sidecar_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(sidecar_filepath, "w") as f:
        json.dump(sidecar, f, indent=4)   

def extract_channels_from_merged_nifti(merged_nifti: nib.Nifti1Image, num_channels: int, return_as: str = "nib") -> list[nib.Nifti1Image | np.ndarray | ants.ANTsImage]:
    """
    Extract channels from a merged NIfTI image.
    
    Args:
        merged_nifti: Merged NIfTI image.
        num_channels: Number of channels to extract.
        return_as: Return the channels as "nib" (NIfTI images) or "np" (NumPy arrays) or "ants" (ANTs images).
    """
    if num_channels <= 0:
        raise ValueError("Number of channels must be greater than 0.")
    if num_channels == 1:
        return [merged_nifti]
    if return_as not in ["nib", "np", "ants"]:
        raise ValueError("Return type must be 'nib' (NIfTI images), 'np' (NumPy arrays), or 'ants' (ANTs images).")
    
    origin = merged_nifti.affine[:3, 3].tolist()
    spacing = merged_nifti.header.get_zooms()
    direction = merged_nifti.affine[:3, :3].tolist()
    
    channels = []
    channel_data = merged_nifti.get_fdata()
    channel_data = np.squeeze(channel_data)
    if return_as == "np":
        for i in range(num_channels):
            channel_data_i = channel_data[..., i]
            channels.append(channel_data_i)
    # if return_as == "nib":
    else:
        for i in range(num_channels):
            channel_data_i = channel_data[..., i]
            channel_nib = nib.Nifti1Image(channel_data_i, affine=merged_nifti.affine)
            if return_as == "nib":
                channels.append(channel_nib)
            elif return_as == "ants":
                channels.append(convert_nibabel_to_ants(channel_nib))

    return channels

def merge_prob_maps(prob_maps: list[ants.ANTsImage | nib.Nifti1Image | np.ndarray]) -> np.ndarray:
    """
    ![Not using it currently] Merge probability maps into hard segmentation.
    """
    if not prob_maps:
        print("Warning: The list of probability maps is empty.")
        return np.array([])

    # Ensure all probability maps have the same shape
    shape = prob_maps[0].shape
    for i, prob_map in enumerate(prob_maps):
        if prob_map.shape != shape:
            raise ValueError(f"Probability map at index {i} has shape {prob_map.shape}, which is different from the first map's shape {shape}.")
        if isinstance(prob_map, ants.ANTsImage):
            prob_maps[i] = prob_map.numpy()
        elif isinstance(prob_map, nib.Nifti1Image):
            prob_maps[i] = prob_map.get_fdata(dtype=np.float32)

    num_maps = len(prob_maps)
    segmentation_map = np.zeros(shape, dtype=np.uint8)  # Initialize with background label 0

    # Stack the probability maps along a new dimension for easier comparison
    stacked_probabilities = np.stack(prob_maps, axis=-1)

    # Find the index of the maximum probability along the last axis (the maps)
    max_indices = np.argmax(stacked_probabilities, axis=-1)

    # Assign labels based on the index of the maximum probability
    for i in range(num_maps):
        segmentation_map[max_indices == i] = i + 1  # Assign label i+1 (1-based for classes)

    return segmentation_map

def convert_nibabel_to_ants(nibabel_image: nib.Nifti1Image) -> ants.ANTsImage:
    """
    Convert a NIfTI image to an ANTs image.
    """
    if not isinstance(nibabel_image, nib.Nifti1Image):
        raise ValueError("Input image must be a NIfTI image.")
    origin = nibabel_image.affine[:3, 3].tolist()
    spacing = nibabel_image.header.get_zooms()
    direction = nibabel_image.affine[:3, :3].tolist()
    
    return ants.from_numpy(nibabel_image.get_fdata(), origin=origin, direction=direction, spacing=spacing)

def convert_ants_to_nibabel(ants_image: ants.ANTsImage) -> nib.Nifti1Image:
    """
    Convert an ANTs image to a NIfTI image.
    """
    if not isinstance(ants_image, ants.ANTsImage):
        raise ValueError("Input image must be an ANTs image.")
    ants_image_data: np.ndarray = ants_image.numpy()
    
    # Get spacing, origin, and direction (ANTs uses LPS, while Nibabel uses RAS)
    spacing = np.array(ants_image.spacing)
    origin = np.array(ants_image.origin)
    direction = np.array(ants_image.direction)
    
    # Construct the affine matrix
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    
    return nib.Nifti1Image(ants_image_data, affine=affine)