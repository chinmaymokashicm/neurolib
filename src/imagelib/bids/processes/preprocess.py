"""
Unitary processes for preprocessing pipelines. 
"""
from ..pipeline import get_new_pipeline_derived_filename
from .helper import save_sidecar

from pathlib import Path, PosixPath
from typing import Optional

import ants
from antspynet.utilities import brain_extraction
from bids import BIDSLayout

def n4_bias_field_correction(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False) -> Optional[None]:
    """
    Execute N4 bias field correction.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    BIDS_DESC_ENTITY: str = "n4bfc"
    try:
        input_nii: ants.ANTsImage = ants.image_read(input_filepath)
        n4_bfc: ants.ANTsImage = ants.n4_bias_field_correction(input_nii, shrink_factor=4, convergence={'iters': [50, 50, 30, 20], 'tol': 1e-7})
        n4_bfc_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".nii.gz")
        
        if not overwrite and Path(n4_bfc_path).exists():
            print(f"File {n4_bfc_path} already exists. Skipping.")
            return BIDS_DESC_ENTITY
        Path(n4_bfc_path).parent.mkdir(parents=True, exist_ok=True)
        n4_bfc.to_filename(n4_bfc_path)
        
        sidecar: dict = {
            "Space": "MNI152NLin2009cAsym",
            "SkullStripped": True,
            "Cropped": True,
            "Resampled": True,
            "N4BiasFieldCorrection": True,
            "Resolutions": {
                "Original": input_nii.shape,
                "N4BiasFieldCorrection": n4_bfc.shape
            },
            "N4BiasFieldCorrection": {
                "BSplineFittingDistance": 300,
                "ShrinkFactor": 4,
                "Iterations": [50, 50, 30, 20]
            },
            "Steps": [
                "N4 bias field correction using ANTsPyNet"
            ]
        }
        sidecar_filepath: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".json")
        save_sidecar(sidecar, sidecar_filepath)
        return BIDS_DESC_ENTITY
    except Exception as e:
        print(f"Failed to run N4 bias field correction for {input_filepath}: {e}")
        sidecar: dict = {
            "Space": "MNI152NLin2009cAsym",
            "SkullStripped": False,
            "Cropped": False,
            "Resampled": False,
            "Resolutions": {
                "Original": input_nii.shape
            },
            "N4BiasFieldCorrection": {
                "BSplineFittingDistance": 300,
                "ShrinkFactor": 4,
                "Iterations": [50, 50, 30, 20],
                "Error": str(e)
            },
            "Steps": [
                "N4 bias field correction"
            ]
        }
        sidecar_filepath: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".json")
        save_sidecar(sidecar, sidecar_filepath)

def brain_extraction_antspynet(input_filepath: str | PosixPath, layout: BIDSLayout, pipeline_name: str, overwrite: bool = False, modality: str = "t1") -> None:
    """
    Execute brain extraction using ANTsPyNet.
    """
    input_filepath: str = str(input_filepath)
    if not input_filepath.endswith((".nii.gz", ".nii")):
        raise ValueError(f"Expected a NIfTI file, got {input_filepath}")
    BIDS_DESC_ENTITY: str = "brainExtract"
    try:
        input_nii: ants.ANTsImage = ants.image_read(input_filepath)
        prob_brain_mask = brain_extraction(input_nii, modality=modality)
        prob_brain_mask_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, "probBrainMask", ".nii.gz")
        if not overwrite and Path(prob_brain_mask_path).exists():
            print(f"File {prob_brain_mask_path} already exists. Skipping.")
            return BIDS_DESC_ENTITY
        Path(prob_brain_mask_path).parent.mkdir(parents=True, exist_ok=True)
        prob_brain_mask.to_filename(prob_brain_mask_path)
        brain_extract = ants.mask_image(input_nii, prob_brain_mask)
        brain_extract_path = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".nii.gz")

        brain_extract.to_filename(brain_extract_path)
        
        sidecar: dict = {
            "Space": "MNI152NLin2009cAsym",
            "BrainExtracted": True,
            "Resampled": False,
            "Resolutions": {
                "Original": input_nii.shape,
                "BrainExtracted": brain_extract.shape
            },
            "Steps": [
                "Brain extracted using ANTsPyNet"
            ]
        }
        sidecar_filepath: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".json")
        save_sidecar(sidecar, sidecar_filepath)
        return BIDS_DESC_ENTITY
    except Exception as e:
        print(f"Failed to run brain extraction for {input_filepath}: {e}")
        sidecar: dict = {
            "Space": "MNI152NLin2009cAsym",
            "BrainExtracted": False,
            "Resampled": False,
            "Resolutions": {
                "Original": input_nii.shape
            },
            "Error": str(e),
            "Steps": [
                "Brain extraction using ANTsPyNet"
            ]
        }
        sidecar_filepath: str = get_new_pipeline_derived_filename(input_filepath, layout, pipeline_name, BIDS_DESC_ENTITY, ".json")
        save_sidecar(sidecar, sidecar_filepath)