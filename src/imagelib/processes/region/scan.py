"""
BIDS Processes that are used individual scans/files to generate region-wise summaries.
"""
from ...processes import register
from ...bids.pipeline import get_new_pipeline_derived_filename
from ...bids.processes.constants import *
from ...bids.processes.base import BIDSProcessSummarySidecar, BIDSProcessResults
from ...bids.processes.helper import merge_prob_maps, convert_nibabel_to_ants, convert_ants_to_nibabel, extract_channels_from_merged_nifti

from pathlib import Path, PosixPath
from typing import Optional
from enum import Enum

import ants
from antspynet.utilities import deep_atropos
from bids import BIDSLayout
from bids.layout import parse_file_entities
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img, load_img
from nilearn.input_data import NiftiLabelsMasker
from skimage import measure

class ParcellationAtlas(str, Enum):
    HARVARD_OXFORD = "harvardOxford"
    DESIKAN_KILLIANY_TOURVILLE = "desikanKillianyTourville"

@BIDSProcessSummarySidecar.execute_process
def cortical_thickness_by_region(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, atlas_name: ParcellationAtlas = ParcellationAtlas.HARVARD_OXFORD, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessResults]:
    """
    Generate cortical thickness values by region of parcellations.
    
    Steps:
    1. Ensure input filepath is a cortical thickness map - enforce this.
    2. Load the parcellation image and sidecar.
    3. Calculate cortical thickness per region.
    4. Save the cortical thickness per region stats to a CSV file.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        atlas_name (ParcellationAtlas, optional): Parcellation atlas name. Defaults to ParcellationAtlas.HARVARD_OXFORD.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
    
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a cortical thickness map - enforce this
    input_filepath_entities = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_CORTICAL_THICKNESS
    input_filepath_entities["suffix"] = "thick"
    input_filepath_entities["extension"] = ".nii.gz"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        print(f"Could not find cortical thickness map for {input_filepath}")
        
    # Check if atlas_name is valid
    if atlas_name not in ParcellationAtlas:
        raise ValueError(f"Invalid atlas name: {atlas_name}. Valid options are: {', '.join([atlas.value for atlas in ParcellationAtlas])}")
    
    suffix: str = "stats"
    # Calculate cortical thickness per region
    parcellation_entities: dict = parse_file_entities(input_filepath)
    
    if atlas_name == ParcellationAtlas.HARVARD_OXFORD:
        parcellation_entities["desc"] = BIDS_DESC_ENTITY_PARCELLATION_HARVARD_OXFORD
        bids_desc_entity_cortical_thickness_stats: str = BIDS_DESC_ENTITY_CORTICAL_THICKNESS_REGION_STATS + ParcellationAtlas.HARVARD_OXFORD[0].capitalize() + ParcellationAtlas.HARVARD_OXFORD[1:]
    elif atlas_name == ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE:
        parcellation_entities["desc"] = BIDS_DESC_ENTITY_PARCELLATION_DKT
        bids_desc_entity_cortical_thickness_stats: str = BIDS_DESC_ENTITY_CORTICAL_THICKNESS_REGION_STATS + ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE[0].capitalize() + ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE[1:]
    
    cortical_thickness_region_stats_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, bids_desc_entity_cortical_thickness_stats, ".csv", suffix=suffix)
    if not overwrite and Path(cortical_thickness_region_stats_path).exists():
        print(f"Skipping {input_filepath} as {cortical_thickness_region_stats_path} already exists.")
        return None
    
    parcellation_entities["suffix"] = "label"
    parcellation_img_path: str = layout.get(return_type="file", **parcellation_entities)[0]
    parcellation_img: nib.Nifti1Image = load_img(parcellation_img_path)
    parcellation_sidecar_path: str = parcellation_img_path.replace(".nii.gz", ".json")
    
    cortical_thickness_img: nib.Nifti1Image = load_img(input_filepath)
    cortical_thickness_data: np.ndarray = cortical_thickness_img.get_fdata()
    
    parcellation_img = resample_to_img(parcellation_img, cortical_thickness_img, interpolation="nearest", force_resample=True, copy_header=True)
    parcellation_data: np.ndarray = parcellation_img.get_fdata()
    
    if atlas_name == ParcellationAtlas.HARVARD_OXFORD:    
        parcellation_sidecar: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar.from_file(parcellation_sidecar_path)
        parcellation_labels: list[str] = parcellation_sidecar.processing[0]["Atlas"]["Labels"]
        
        regional_cortical_thickness_metrics: list[dict] = []
        
        for i, label in enumerate(parcellation_labels, start=1):
            label_mask = parcellation_data == i
            label_thickness = cortical_thickness_data[label_mask]
            mean_thickness = np.mean(label_thickness)
            std_thickness = np.std(label_thickness)
            regional_cortical_thickness_metrics.append({
                "Region": label,
                "mean_cortical_thickness": mean_thickness,
                "std_cortical_thickness": std_thickness
            })
        
    elif atlas_name == ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE:
        parcellation_sidecar: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar.from_file(parcellation_sidecar_path)
        parcellation_labels: list[dict[str, str]] = parcellation_sidecar.processing[0]["Atlas"]["Labels"]
        if not isinstance(parcellation_labels, list):
            print(f"Expected a list of dictionaries, got {type(parcellation_labels)}")
        if len(parcellation_labels) != len(np.unique(parcellation_data)) - 1: # -1 because the first label is 0, which is removed in the sidecar
            print(f"Number of parcellation labels in atlas for {input_filepath}: {len(parcellation_labels)}. Number of parcellations found in the image: {len(np.unique(parcellation_data))} - MISMATCH")
        
        regional_cortical_thickness_metrics: list[dict] = []
        
        for row in parcellation_labels:
            if not isinstance(row, dict):
                raise ValueError(f"Expected a dictionary, got {type(row)}")
            _, label, name = row["region"], int(row["label"]), row["name"]
            label_mask = parcellation_data == label
            label_thickness = cortical_thickness_data[label_mask]
            mean_thickness = np.mean(label_thickness)
            std_thickness = np.std(label_thickness)
            regional_cortical_thickness_metrics.append({
                "Region": name,
                "mean_cortical_thickness": mean_thickness,
                "std_cortical_thickness": std_thickness
            })
    
    df_cortical_thickness_region_stats: pd.DataFrame = pd.DataFrame(regional_cortical_thickness_metrics)
    df_cortical_thickness_region_stats.to_csv(cortical_thickness_region_stats_path, index=False)
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": cortical_thickness_img.shape},
        output={"path": cortical_thickness_region_stats_path},
        processing=[{"CorticalThicknessByRegion": {"Parcellation": parcellation_img_path}}],
        steps=["Calculate cortical thickness per region of parcellation"],
        status="success",
        metrics=regional_cortical_thickness_metrics,
    )
    
    
@register
@BIDSProcessSummarySidecar.execute_process
def tissue_segment_stats_per_region(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, atlas_name: ParcellationAtlas = ParcellationAtlas.HARVARD_OXFORD, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessResults]:
    """
    Generate statistics of tissue segments by region of parcellations (ROIs).
    
    Steps:
    1, Ensure input filepath is a multi-channel tissue segmentation map - enforce this.
    2. Load the parcellation image and sidecar.
    3. Calculate the following statistics-
        - Signal intensity per region - use brain extracted T1w image
        - Cortical surface area per region - use gray matter segmentation
        - Volume per region per segment - use CSF, gray matter, white matter segmentations
        
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        atlas_name (ParcellationAtlas, optional): Parcellation atlas name. Defaults to ParcellationAtlas.HARVARD_OXFORD.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    
    seg_filepath: str = str(input_filepath)
    if not seg_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {seg_filepath}")
    
    # * Input filepath has to be a multi-channel tissue segmentation map - enforce this
    seg_filepath_entities = parse_file_entities(seg_filepath)
    seg_filepath_entities["desc"] = BIDS_DESC_ENTITY_TISSUE_SEGMENT
    seg_filepath_entities["suffix"] = "seg"
    seg_filepath_entities["extension"] = ".nii.gz"
    try:
        seg_filepath = layout.get(return_type="file", **seg_filepath_entities)[0]
    except IndexError:
        print(f"Could not find tissue segmentation map for {seg_filepath}")
        
    # Check if atlas_name is valid
    if atlas_name not in ParcellationAtlas:
        raise ValueError(f"Invalid atlas name: {atlas_name}. Valid options are: {', '.join([atlas.value for atlas in ParcellationAtlas])}")
    
    # Calculate gray matter surface area per region
    parcellation_entities: dict = parse_file_entities(seg_filepath)
    
    if atlas_name == ParcellationAtlas.HARVARD_OXFORD:
        parcellation_entities["desc"] = BIDS_DESC_ENTITY_PARCELLATION_HARVARD_OXFORD
        bids_desc_entity_tissue_segment_region_stats: str = BIDS_DESC_ENTITY_SEGMENT_REGION_STATS + ParcellationAtlas.HARVARD_OXFORD[0].capitalize() + ParcellationAtlas.HARVARD_OXFORD[1:]
    elif atlas_name == ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE:
        parcellation_entities["desc"] = BIDS_DESC_ENTITY_PARCELLATION_DKT
        bids_desc_entity_tissue_segment_region_stats: str = BIDS_DESC_ENTITY_SEGMENT_REGION_STATS + ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE[0].capitalize() + ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE[1:]
        
    suffix: str = "stats"
    tissue_segment_stats_region_path: str = get_new_pipeline_derived_filename(seg_filepath, layout, pipeline_name, bids_desc_entity_tissue_segment_region_stats, ".csv", suffix=suffix)
    if not overwrite and Path(tissue_segment_stats_region_path).exists():
        print(f"Skipping {seg_filepath} as {tissue_segment_stats_region_path} already exists.")
        return None
    
    # Get the parcellation image
    parcellation_entities["suffix"] = "label"
    parcellation_img_path: str = layout.get(return_type="file", **parcellation_entities)[0]
    parcellation_sidecar_path: str = parcellation_img_path.replace(".nii.gz", ".json")
    parcellation_img: nib.Nifti1Image = load_img(parcellation_img_path)
    parcellation_data: np.ndarray = parcellation_img.get_fdata()
    
    # Get the tissue segmentation images
    tissue_segmentation_img: nib.Nifti1Image = load_img(seg_filepath)
    _, _, _, segmentation = extract_channels_from_merged_nifti(tissue_segmentation_img, num_channels=4, return_as="ants")
    csf_seg, gray_matter_seg, white_matter_seg = segmentation == 1, segmentation == 2, segmentation == 3
    csf_data, gray_matter_data, white_matter_data = csf_seg.numpy(), gray_matter_seg.numpy(), white_matter_seg.numpy()
    
    # Get brain extracted T1w image (to calculate signal intensity per region)
    brain_extract_entities: dict = parse_file_entities(seg_filepath)
    brain_extract_entities["desc"] = BIDS_DESC_ENTITY_BRAIN_EXTRACT
    brain_extract_entities["suffix"] = "T1w"
    brain_extract_entities["extension"] = ".nii.gz"
    brain_extract_img_path: str = layout.get(return_type="file", **brain_extract_entities)[0]
    brain_extract_img: nib.Nifti1Image = load_img(brain_extract_img_path)
        
    # stats: dict[str, dict[str, float]] = {}
    stats: list[dict[str, float]] = []
    
    # Calculate signal intensity per region
    masker = NiftiLabelsMasker(labels_img=parcellation_img, standardize=False)
    signal_intensity_per_region = masker.fit_transform(brain_extract_img)
    
    if atlas_name == ParcellationAtlas.HARVARD_OXFORD:
        parcellation_sidecar: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar.from_file(parcellation_sidecar_path)
        parcellation_labels: list[str] = parcellation_sidecar.processing[0]["Atlas"]["Labels"]
        
        for i, label in enumerate(parcellation_labels, start=1): # Start from 1 because the first label is 0 (background)
            signal_intensity = signal_intensity_per_region[0, i - 1]
            
            label_mask = parcellation_data == i
            
            label_csf = csf_data * label_mask
            label_gray_matter = gray_matter_data * label_mask
            label_white_matter = white_matter_data * label_mask
            
            # Calculate cortical surface area
            try:
                verts, faces, _, _ = measure.marching_cubes(label_gray_matter, level=0.5)
                cortical_surface_area = measure.mesh_surface_area(verts, faces)
            except ValueError:
                cortical_surface_area = 0
            
            # Calculate volume per region per segment
            csf_volume = np.sum(label_csf)
            gray_matter_volume = np.sum(label_gray_matter)
            white_matter_volume = np.sum(label_white_matter)
            
            stats.append(
                {
                    "Region": label,
                    "signal_intensity": float(signal_intensity),
                    "cortical_surface_area": float(cortical_surface_area),
                    "csf_volume": float(csf_volume),
                    "gray_matter_volume": float(gray_matter_volume),
                    "white_matter_volume": float(white_matter_volume)
                }
            )
            
    elif atlas_name == ParcellationAtlas.DESIKAN_KILLIANY_TOURVILLE:
        parcellation_sidecar: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar.from_file(parcellation_sidecar_path)
        parcellation_labels: list[dict[str, str]] = parcellation_sidecar.processing[0]["Atlas"]["Labels"]
        if not isinstance(parcellation_labels, list):
            print(f"Expected a list of dictionaries, got {type(parcellation_labels)}")
        if len(parcellation_labels) != len(np.unique(parcellation_data)) - 1:
            print(f"Number of parcellation labels in atlas for {seg_filepath}: {len(parcellation_labels)}. Number of parcellations found in the image: {len(np.unique(parcellation_data))} - MISMATCH")

        for i, row in enumerate(parcellation_labels):
            if not isinstance(row, dict):
                raise ValueError(f"Expected a dictionary, got {type(row)}")
            _, label, name = row["region"], int(row["label"]), row["name"]
            signal_intensity = signal_intensity_per_region[0, i]
            
            label_mask = parcellation_data == label
            
            label_csf = csf_data * label_mask
            label_gray_matter = gray_matter_data * label_mask
            label_white_matter = white_matter_data * label_mask
            
            # Calculate cortical surface area
            try:
                verts, faces, _, _ = measure.marching_cubes(label_gray_matter, level=0.5)
                cortical_surface_area = measure.mesh_surface_area(verts, faces)
            except ValueError:
                cortical_surface_area = 0
            
            # Calculate volume per region per segment
            csf_volume = np.sum(label_csf)
            gray_matter_volume = np.sum(label_gray_matter)
            white_matter_volume = np.sum(label_white_matter)
            
            stats.append(
                {
                    "Region": name,
                    "signal_intensity": float(signal_intensity),
                    "cortical_surface_area": float(cortical_surface_area),
                    "csf_volume": float(csf_volume),
                    "gray_matter_volume": float(gray_matter_volume),
                    "white_matter_volume": float(white_matter_volume)
                }
            )
            
    df_tissue_segment_stats_region: pd.DataFrame = pd.DataFrame(stats)
    df_tissue_segment_stats_region.to_csv(tissue_segment_stats_region_path, index=False)
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": seg_filepath, "resolution": tissue_segmentation_img.shape},
        output={"path": tissue_segment_stats_region_path},
        processing=[{"TissueSegmentStatsByRegion": {"Parcellation": parcellation_img_path}}],
        steps=["Calculate tissue segment stats per region"],
        status="success",
        metrics=stats
    )
