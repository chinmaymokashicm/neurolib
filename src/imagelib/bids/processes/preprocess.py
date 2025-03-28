"""
Unitary processes for preprocessing pipelines. 
"""
from ..pipeline import get_new_pipeline_derived_filename
from .constants import *
from .base import BIDSProcessSummarySidecar, BIDSProcessResults
from ...helpers.data import clean_dict_from_numpy

from pathlib import Path, PosixPath
from typing import Optional

import ants
from antspynet.utilities import brain_extraction
from bids import BIDSLayout

@BIDSProcessSummarySidecar.execute_process
def registration_to_mni(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, mni_template_path: str, overwrite: bool = False) -> Optional[BIDSProcessResults]:
    """
    Execute image registration to MNI space.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        mni_template_path (str): Path to the MNI template.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    suffix: str = "T1w"
    mni_warped_path: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_MNI152, ".nii.gz", suffix=suffix)
    if not overwrite and Path(mni_warped_path).exists():
        print(f"File {mni_warped_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    mni_template: ants.ANTsImage = ants.image_read(mni_template_path)
    
    # Resample the input image to the MNI template
    input_nii: ants.ANTsImage = ants.resample_image_to_target(input_nii, mni_template)
    
    mni_transformation = ants.registration(
        fixed=mni_template,
        moving=input_nii,
        type_of_transform="SyN"
    )
    mni_warped: ants.ANTsImage = ants.apply_transforms(
        fixed=mni_template,
        moving=input_nii,
        transformlist=mni_transformation["fwdtransforms"]
    )
    Path(mni_warped_path).parent.mkdir(parents=True, exist_ok=True)
    mni_warped.to_filename(mni_warped_path)
    
    return BIDSProcessResults(
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": mni_warped_path, "resolution": mni_warped.shape},
        processing=[{"RegistrationToMNI": {"Template": mni_template_path}}],
        steps=["Registration to MNI space using ANTsPyNet"],
        status="success",
        metrics=[
            {"warpedmovout": {"spacing": mni_transformation["warpedmovout"].spacing, "origin": mni_transformation["warpedmovout"].origin, "direction": mni_transformation["warpedmovout"].direction.tolist()}},
            {"warpedfixout": {"spacing": mni_transformation["warpedfixout"].spacing, "origin": mni_transformation["warpedfixout"].origin, "direction": mni_transformation["warpedfixout"].direction.tolist()}},
        ]
    )

@BIDSProcessSummarySidecar.execute_process
def n4_bias_field_correction(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False) -> Optional[BIDSProcessResults]:
    """
    Execute N4 bias field correction.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    suffix: str = "T1w"
    SHRINK_FACTOR: int = 4
    CONVERGENCE: dict = {'iters': [50, 50, 30, 20], 'tol': 1e-7}
    n4_bfc_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_N4BFC, ".nii.gz", suffix=suffix)
    if not overwrite and Path(n4_bfc_path).exists():
        print(f"File {n4_bfc_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    
    n4_bfc: ants.ANTsImage = ants.n4_bias_field_correction(input_nii, shrink_factor=SHRINK_FACTOR, convergence=CONVERGENCE)
    
    Path(n4_bfc_path).parent.mkdir(parents=True, exist_ok=True)
    n4_bfc.to_filename(n4_bfc_path)
    
    return BIDSProcessResults(
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": n4_bfc_path, "resolution": n4_bfc.shape},
        processing=[{"N4BiasFieldCorrection": {"ShrinkFactor": SHRINK_FACTOR, "Convergence": CONVERGENCE}}],
        steps=["N4 bias field correction using ANTsPyNet"],
        status="success",
        metrics=[{"ShrinkFactor": SHRINK_FACTOR, "Convergence": CONVERGENCE}]
    )

@BIDSProcessSummarySidecar.execute_process
def brain_extraction_antspynet(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, modality: str = "t1") -> Optional[BIDSProcessResults]:
    """
    Execute brain extraction using ANTsPyNet.
    
    Args:
        input_filepath (str | PosixPath): Input NIfTI file path.
        layout (BIDSLayout): BIDSLayout object.
        pipeline_name (str): Name of the pipeline.
        overwrite (bool, optional): Overwrite existing files. Defaults to False.
        modality (str, optional): Modality of the image. Defaults to "t1".
        
    Returns:
        Optional[BIDSProcessResults]: Results of the process. If output file already exists, returns None.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    suffix: str = "T1w"
    brain_extract_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_BRAIN_EXTRACT, ".nii.gz", suffix=suffix)
    if not overwrite and Path(brain_extract_path).exists():
        print(f"File {brain_extract_path} already exists. Skipping.")
        return None
    input_nii: ants.ANTsImage = ants.image_read(input_filepath)
    prob_brain_mask_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY_PROB_BRAIN_MASK, ".nii.gz", suffix="mask")
    prob_brain_mask = brain_extraction(input_nii, modality=modality)
    Path(prob_brain_mask_path).parent.mkdir(parents=True, exist_ok=True)
    prob_brain_mask.to_filename(prob_brain_mask_path)
    brain_extract = ants.mask_image(input_nii, prob_brain_mask)

    brain_extract.to_filename(brain_extract_path)
    
    return BIDSProcessResults(
        input={"path": input_filepath, "resolution": input_nii.shape},
        output={"path": brain_extract_path, "resolution": brain_extract.shape},
        processing=[{"BrainExtraction": {"Modality": modality}}],
        steps=["Brain extraction using ANTsPyNet"],
        status="success",
        metrics=[{"Modality": modality}]
    )