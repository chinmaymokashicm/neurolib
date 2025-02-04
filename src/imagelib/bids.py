"""
Convert DICOM dataset to BIDS dataset. The organization of the DICOM dataset is variable.
Use of the tool dcm2bids as the package for the conversion.

Steps of conversion-
0. Installation of dcm2bids and dcm2niix
1. Create a scaffolding of the BIDS dataset
2. Migrate the DICOM dataset to sourcedata/ subdirectory (per subject and session)
3. Use dcm2bids_helper to create example sidecar json files
3. Build the configuration file for dcm2bids - use from sidecar json files
4. Run dcm2bids with each session of the DICOM dataset having a unique participant ID and session ID
"""

import shutil, subprocess, json
from pathlib import Path, PosixPath
from typing import Optional

from pydantic import BaseModel
from rich.progress import track

class ParticipantMapping(BaseModel):
    """
    Map existing subject ID and session ID to BIDS participant ID and session ID.
    """
    subject_id: str
    session_id: str
    participant_id: str
    session_participant_id: str
    dicom_subdir: str

def read_dicom2bids_mapping(mapping_file: PosixPath) -> list[ParticipantMapping]:
    mappings: list[ParticipantMapping] = []
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
    
    for subject_id, sessions in mapping.items():
        for session_id, participant_info in sessions.items():
            participant_id: str = participant_info["participant_id"]
            participant_session_id: str = participant_info["session_id"]
            mappings.append(
                ParticipantMapping(
                    subject_id=subject_id,
                    session_id=session_id,
                    participant_id=participant_id,
                    session_participant_id=participant_session_id,
                    dicom_subdir=participant_info["dicom_subdir"]
                )
            )
    
    return mappings
    
class DICOMToBIDSConvertor(BaseModel):
    bids_root: PosixPath
    dicom_root: PosixPath
    participant_mappings: list[ParticipantMapping]
    
    def create_bids_scaffolding(self):
        """
        Create the scaffolding of the BIDS dataset.
        """
        # Create directory if not exists
        if not self.bids_root.exists():
            self.bids_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dcm2bids_scaffold", "-o", str(self.bids_root)], check=True)
        
    def migrate_dicom_data(self, symlink: bool = True, sample: bool = True):
        # Migrate a small subset if sample is True
        if sample:
            self.participant_mappings = self.participant_mappings[:2]
        
        for participant_mapping in track(self.participant_mappings):
            # Create participant and session directories
            participant_dir = self.bids_root / "sourcedata" / participant_mapping.participant_id
            session_dir = participant_dir / participant_mapping.session_participant_id
            if not participant_dir.exists():
                participant_dir.mkdir(parents=True, exist_ok=True)
            if not session_dir.exists():
                session_dir.mkdir(parents=True, exist_ok=True)

            # Migrate DICOM data to sourcedata/ subdirectory
            dicom_dir = self.dicom_root / participant_mapping.dicom_subdir
            shutil.copytree(dicom_dir, session_dir, dirs_exist_ok=True, symlinks=symlink)
            
    def run_dcm2bids_helper(self, subject_id: str, session_id: str, output_dir: PosixPath) -> None:
        """
        Run dcm2bids_helper to create example sidecar json files.
        """
        participant_mapping = next((mapping for mapping in self.participant_mappings if mapping.subject_id == subject_id and mapping.session_id == session_id), None)
        dicom_subdir_full_path: PosixPath = self.dicom_root / participant_mapping.dicom_subdir
        subprocess.run(["dcm2bids_helper", "-d", str(dicom_subdir_full_path), "-o", str(output_dir)], check=True)
