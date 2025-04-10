"""
Unitary BIDS processes for segmentation pipelines.
"""
from ..bids.pipeline import get_new_pipeline_derived_filename
from ..bids.processes.constants import *
from ..bids.processes.base import BIDSProcessSummarySidecar, BIDSProcessResults
from ..bids.processes.helper import merge_prob_maps, convert_nibabel_to_ants, convert_ants_to_nibabel, extract_channels_from_merged_nifti

from pathlib import Path, PosixPath
from typing import Optional
import logging

import ants
from antspynet.utilities import deep_atropos, desikan_killiany_tourville_labeling
from bids import BIDSLayout
from bids.layout import parse_file_entities
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.image import resample_to_img, load_img
from nilearn.input_data import NiftiLabelsMasker

logger = logging.getLogger(__name__)

@BIDSProcessSummarySidecar.execute_process
def atropos_tissue_segmentation(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, process_id: Optional[str] = None,  process_exec_id: Optional[str] = None,pipeline_id: Optional[str] = None) -> Optional[BIDSProcessSummarySidecar]:
    """
    Execute tissue segmentation using ANTs Finite Mixture Modeling.
    
    Steps:
    1. Enforces that the input file is a whole MNI-registered brain file.
    2. Enforces that the input mask file is a brain mask file.
    3. Runs Atropos tissue segmentation.
    4. Saves the probability maps and the segmentation as a multi-channel NIfTI file.
    5. Calculates volume metrics of the tissue segmentations.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process Execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a MNI-registered whole image - enforce this
    input_filepath_entities: dict = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_MNI152
    input_filepath_entities["suffix"] = "T1w"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        logger.error(f"Could not find MNI-registered file for {input_filepath}. Skipping.")
    
    suffix: str = "seg"
    tissue_segment_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_TISSUE_SEGMENT, ".nii.gz", suffix=suffix)
    if not overwrite and Path(tissue_segment_path).exists():
        logger.error(f"File {tissue_segment_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    input_mask_entities: dict = parse_file_entities(input_filepath)
    input_mask_entities["desc"] = BIDS_DESC_ENTITY_PROB_BRAIN_MASK
    input_mask_entities["suffix"] = "mask"
    input_nii_mask_path: str = layout.get(return_type="file", **input_mask_entities)[0]
    input_nii_mask: ants.ANTsImage = ants.image_read(input_nii_mask_path)
    seg = ants.atropos(
        a=input_nii,
        m="[0.2,1x1x1]",
        c="[5,0]", 
        i="kmeans[3]",
        x=input_nii_mask
    )
    csf, gray_matter, white_matter = seg["probabilityimages"][0], seg["probabilityimages"][1], seg["probabilityimages"][2]
    prob_maps = ants.merge_channels([csf, gray_matter, white_matter, seg["segmentation"]])
    
    Path(tissue_segment_path).parent.mkdir(parents=True, exist_ok=True)
    
    # * Save the probability maps as multi-channel NIfTI: Will need extraction of individual channels
    prob_maps.to_filename(tissue_segment_path)
    
    # Calculate volume metrics of the tissue segmentations
    label_stats_all: list[dict] = []
    for idx, segment in enumerate(seg["probabilityimages"]):
        try:
            df_label_stats: pd.DataFrame = ants.label_stats(input_nii, segment)
            region_name: str = ["CSF", "Gray Matter", "White Matter", "Segmentation"][idx]
            label_stats: dict = df_label_stats[df_label_stats["LabelValue"] == 1.0].to_dict(orient="records")[0]
            label_stats_all.append({region_name: label_stats})
        except Exception as e:
            # logger.error(f"Failed to calculate label stats for {segment} in {input_filepath}: {e}")
            logger.error(f"Failed to calculate label stats for {input_filepath}: {e}")
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": tissue_segment_path, "resolution": prob_maps.shape},
        processing=[{"Atropos": {"Initialization Method": "kmeans[3]", "Convergence": "[5,0]", "Prior": "[0.2,1x1x1]"}}],
        steps=["Tissue segmentation using ANTs Finite Mixture Modeling - Atropos"],
        status="success",
        metrics=label_stats_all
    )
        
@BIDSProcessSummarySidecar.execute_process
def deep_atropos_tissue_segmentation(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessResults]:
    """
    Execute tissue segmentation using ANTs Deep Learning (deep_atropos).
    
    Steps:
    1. Enforces that the input file is a MNI-registered brain extracted file.
    2. Runs Deep Atropos tissue segmentation.
    3. Saves the probability maps and the segmentation as a multi-channel NIfTI file.
    4. Calculates volume metrics of the tissue segmentations.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process Execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a MNI-registered brain extracted file - enforce this
    input_filepath_entities: dict = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_MNI152
    input_filepath_entities["suffix"] = "T1w"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        logger.error(f"Could not find brain extracted file for {input_filepath}. Skipping.")
    
    suffix: str = "deepseg"
    tissue_segment_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_TISSUE_SEGMENT, ".nii.gz", suffix=suffix)
    if not overwrite and Path(tissue_segment_path).exists():
        logger.error(f"File {tissue_segment_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    deep_atropos_output = deep_atropos(input_nii, do_preprocessing=False)
    deep_atropos_output[deep_atropos_output == 4] = 3
    csf = deep_atropos_output["probability_images"][1]
    gray_matter = deep_atropos_output["probability_images"][2]
    white_matter = deep_atropos_output["probability_images"][3] + deep_atropos_output["probability_images"][4]
    tissue_segment: ants.ANTsImage = ants.merge_channels([csf, gray_matter, white_matter, deep_atropos_output["segmentation_image"]])
    
    Path(tissue_segment_path).parent.mkdir(parents=True, exist_ok=True)
    tissue_segment.to_filename(tissue_segment_path)
    
    # Calculate volume metrics of the tissue segmentations
    label_stats_all: list[dict] = []
    for segment in deep_atropos_output["probability_images"]:
        try:
            df_label_stats: pd.DataFrame = ants.label_stats(input_nii, segment)
            label_stats: dict = df_label_stats[df_label_stats["LabelValue"] == 1.0].to_dict(orient="records")[0]
            label_stats_all.append(label_stats)
        except Exception as e:
            # logger.error(f"Failed to calculate label stats for {segment} in {input_filepath}: {e}")
            logger.error(f"Failed to calculate label stats for {input_filepath}: {e}")
            
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": tissue_segment_path, "resolution": tissue_segment.shape},
        processing=[{"DeepAtropos": {"Model": "Deep Atropos in ANTs"}}],
        steps=["Tissue segmentation using ANTs Deep Learning"],
        status="success",
        metrics=label_stats_all
    )

@BIDSProcessSummarySidecar.execute_process
def brain_parcellation_harvard_oxford_nilearn(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, atlas_spec: str, overwrite: bool = False, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessSummarySidecar]:
    """
    Execute brain parcellation using nilearn's fetch_atlas_harvard_oxford.
    
    1. Enforces that the input file is a brain extracted file.
    2. Registers the atlas to the input image.
    3. Resamples the atlas to the input image.
    4. Extracts region signals from the input image.
    5. Saves the process summary sidecar.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        atlas_spec (str): Name of the atlas specification.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process Execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessSummarySidecar]: Process summary sidecar if successful, else None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a brain extracted file - enforce this
    input_filepath_entities: dict = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_BRAIN_EXTRACT
    input_filepath_entities["suffix"] = "T1w"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        logger.error(f"Could not find brain extracted file for {input_filepath}. Skipping.")
    
    suffix: str = "label"
    parcellation_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_PARCELLATION_HARVARD_OXFORD, ".nii.gz", suffix=suffix)
    if not overwrite and Path(parcellation_path).exists():
        logger.error(f"File {parcellation_path} already exists. Skipping.")
        return None
    input_nii: nib.Nifti1Image = load_img(input_filepath)
    atlas = fetch_atlas_harvard_oxford(atlas_spec)
    atlas_img: nib.Nifti1Image = atlas["maps"]
    atlas_labels = atlas["labels"][1:] # Skip the background label
    
    # Register the atlas to the input image
    atlas_img_ants: ants.ANTsImage = convert_nibabel_to_ants(atlas_img)
    input_nii_ants: ants.ANTsImage = convert_nibabel_to_ants(input_nii)
    atlas_to_input = ants.registration(fixed=input_nii_ants, moving=atlas_img_ants, type_of_transform="SyN")
    atlas_img_warped = ants.apply_transforms(fixed=input_nii_ants, moving=atlas_img_ants, transformlist=atlas_to_input["fwdtransforms"])
    
    # Re-convert the warped atlas back to NIfTI
    atlas_img = convert_ants_to_nibabel(atlas_img_warped)
    
    resampled_atlas = resample_to_img(atlas_img, input_nii, interpolation="nearest", force_resample=True, copy_header=True)
    
    # Save the resampled atlas
    Path(parcellation_path).parent.mkdir(parents=True, exist_ok=True)
    resampled_atlas.to_filename(parcellation_path)
    
    masker = NiftiLabelsMasker(labels_img=resampled_atlas, standardize=False)
    
    region_signals = masker.fit_transform(input_filepath)
    region_signals_dict = {atlas_labels[i]: round(float(region_signals[0, i]), 2) for i in range(len(atlas_labels))}
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": parcellation_path, "resolution": resampled_atlas.shape},
        processing=[{"Atlas": {"Name": atlas_spec, "Labels": atlas_labels}}],
        steps=["Brain parcellation using Harvard-Oxford Atlas"],
        status="success",
        metrics=[{"Region Signals": region_signals_dict}]
    )
        
@BIDSProcessSummarySidecar.execute_process
def brain_parcellation_desikan_killiany_tourville(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessResults]:
    """
    Execute brain parcellation using ANTs' Desikan-Killiany-Tourville labeling.
    
    Steps:
    1. Enforces that the input file is a brain extracted file.
    2. Runs Desikan-Killiany-Tourville labeling.
    3. Saves the region labels as a sidecar.
    4. Saves the parcellation as a NIfTI file.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process Execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a brain extracted file - enforce this
    input_filepath_entities: dict = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_BRAIN_EXTRACT
    input_filepath_entities["suffix"] = "T1w"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        logger.error(f"Could not find brain extracted file for {input_filepath}. Skipping.")
    
    suffix: str = "label"
    parcellation_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_PARCELLATION_DKT, ".nii.gz", suffix=suffix)
    if not overwrite and Path(parcellation_path).exists():
        logger.error(f"File {parcellation_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    parc: ants.ANTsImage = desikan_killiany_tourville_labeling(input_nii, do_preprocessing=False, return_probability_images=False, do_lobar_parcellation=False)
    parc_labels_path: PosixPath = Path(ROOT_DIR) / "src" / "imagelib" /  "resources" / "dkt_regions.csv"
    if not parc_labels_path.exists():
        raise FileNotFoundError(f"Could not find Desikan-Killiany-Tourville region labels at {parc_labels_path}")
    df_parc_labels: pd.DataFrame = pd.read_csv(parc_labels_path, dtype="str")
    parc_img_labels: list[str] = [str(int(label)) for label in np.unique(parc.numpy()).tolist()]
    # Keep only inner and outer labels
    df_parc_labels = df_parc_labels[df_parc_labels["label"].isin(parc_img_labels) & df_parc_labels["region"].str.contains("inner|outer")]
    df_parc_labels = df_parc_labels[df_parc_labels["label"] != "0"]
    region_labels: list[dict] = df_parc_labels.to_dict(orient="records")
    Path(parcellation_path).parent.mkdir(parents=True, exist_ok=True)
    parc.to_filename(parcellation_path)
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": parcellation_path, "resolution": parc.shape},
        processing=[{"Atlas": {"Name": "Desikan-Killiany-Tourville", "Labels": region_labels}}],
        steps=["Brain parcellation using Desikan-Killiany-Tourville Atlas"],
        status="success",
        metrics=[{"region_labels": region_labels}]
    )

@BIDSProcessSummarySidecar.execute_process
def kelly_kapowski_cortical_thickness(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, process_id: Optional[str] = None, process_exec_id: Optional[str] = None, pipeline_id: Optional[str] = None) -> Optional[BIDSProcessResults]:
    """
    Execute cortical thickness estimation using ANTs' Kelly-Kapowski algorithm.
    
    Steps:
    1. Enforces that the input file is a tissue segmentation file.
    2. Runs ANTs' Kelly-Kapowski algorithm for cortical thickness estimation.
    3. Registers the cortical thickness image to the brain extracted image.
    4. Saves the cortical thickness image.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        process_id (Optional[str], optional): Process ID. Defaults to None.
        process_exec_id (Optional[str], optional): Process Execution ID. Defaults to None.
        pipeline_id (Optional[str], optional): Pipeline ID. Defaults to None.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    
    # * Input filepath has to be a tissue segmentation file - enforce this
    input_filepath_entities: dict = parse_file_entities(input_filepath)
    input_filepath_entities["desc"] = BIDS_DESC_ENTITY_TISSUE_SEGMENT
    input_filepath_entities["suffix"] = "seg"
    try:
        input_filepath = layout.get(return_type="file", **input_filepath_entities)[0]
    except IndexError:
        logger.error(f"Could not find tissue segmentation file for {input_filepath}. Skipping.")
    
    suffix: str = "thick"
    cortical_thickness_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_CORTICAL_THICKNESS, ".nii.gz", suffix=suffix)
    if not overwrite and Path(cortical_thickness_path).exists():
        logger.error(f"File {cortical_thickness_path} already exists. Skipping.")
        return None
    
    segment_nii: nib.Nifti1Image = load_img(input_filepath)
    _, gray_matter, white_matter, segmentation = extract_channels_from_merged_nifti(segment_nii, num_channels=4, return_as="ants")
    cortical_thickness_img: ants.ANTsImage = ants.kelly_kapowski(
        s=segmentation, g=gray_matter, w=white_matter,
        # gm_label=2, wm_label=3
        )
    
    # # Register the cortical thickness image to the brain extracted image
    brain_extract_entities: dict = parse_file_entities(input_filepath)
    brain_extract_entities["desc"] = BIDS_DESC_ENTITY_BRAIN_EXTRACT
    brain_extract_entities["suffix"] = "T1w"
    brain_extract_img_path: str = layout.get(return_type="file", **brain_extract_entities)[0]
    brain_extract_img: ants.ANTsImage = ants.image_read(brain_extract_img_path)
    
    # ! Observation - the output image of the KK algorithm has a different spacing - the direction is 
    # ! being coerced to identity when saving. This is causing the registration to fail. This is a fix that
    # ! seems to be working for now.
    # RAS direction
    ras_direction: np.ndarray = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cortical_thickness_img.set_direction(ras_direction)
    
    # cortical_thickness_img_to_brain_extract = ants.registration(fixed=brain_extract_img, moving=cortical_thickness_img, type_of_transform="SyN")
    cortical_thickness_img_to_brain_extract = ants.registration(fixed=brain_extract_img, moving=cortical_thickness_img, type_of_transform="Rigid")
    cortical_thickness_img_warped = ants.apply_transforms(fixed=brain_extract_img, moving=cortical_thickness_img, transformlist=cortical_thickness_img_to_brain_extract["fwdtransforms"])
    
    Path(cortical_thickness_path).parent.mkdir(parents=True, exist_ok=True)
    cortical_thickness_img_warped.to_filename(cortical_thickness_path)
    
    return BIDSProcessResults(
        process_id=process_id,
        process_exec_id=process_exec_id,
        pipeline_id=pipeline_id,
        input={"path": input_filepath, "resolution": segment_nii.shape},
        output={"path": cortical_thickness_path, "resolution": cortical_thickness_img_warped.shape},
        processing=[{"KellyKapowski": {"Algorithm": "ANTs' Kelly-Kapowski"}}],
        steps=["Cortical Thickness Estimation using ANTs' Kelly-Kapowski Algorithm"],
        status="success",
        metrics=[]
    )