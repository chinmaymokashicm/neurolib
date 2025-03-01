"""
Convert DICOM dataset to BIDS dataset. The organization of the DICOM dataset is variable.
Use of the tool dcm2bids as the package for the conversion.

Steps of conversion-
0. Installation of dcm2bids and dcm2niix
1. Create a scaffolding of the BIDS dataset
2. Create a 1:1 mapping of existing subject ID and session ID to BIDS participant ID and session ID (to be done by the user - not here)
2. Migrate the DICOM dataset to sourcedata/ subdirectory (per subject and session)
3. Use dcm2bids_helper to create example sidecar json files (optional)
3. Build the configuration file for dcm2bids - use from sidecar json files (non-programmatically)
4. Run dcm2bids with each session of the DICOM dataset having a unique participant ID and session ID
"""

from .helpers.file import copy_as_symlinks
from .dicom import Participant, Participants

import shutil, subprocess
from pathlib import Path, PosixPath
import multiprocessing

from pydantic import BaseModel, FilePath, DirectoryPath, field_validator
from rich.progress import track
    
class DICOMToBIDSConvertor(BaseModel):
    bids_root: DirectoryPath
    dicom_root: DirectoryPath
    participants: Participants
    config: FilePath
    
    @field_validator("bids_root", mode="before")
    def create_bids_root(cls, value):
        if not value.exists():
            value.mkdir(parents=True, exist_ok=True)
        return value
    
    def create_bids_scaffolding(self):
        """
        Create the scaffolding of the BIDS dataset.
        """
        # Create directory if not exists
        if not self.bids_root.exists():
            self.bids_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dcm2bids_scaffold", "-o", str(self.bids_root)], check=False)
        
    def migrate_dicom_data(self, symlink: bool = True, sample: bool = True):
        # Migrate a small subset if sample is True
        end: int = 2 if sample else len(self.participants.participants)
        
        for participant in self.participants.participants[:end]:
            # Create participant and session directories
            participant_dir = self.bids_root / "sourcedata" / participant.subject_id
            if not participant_dir.exists():
                participant_dir.mkdir(parents=True, exist_ok=True)
            
            for session_info in track(participant.sessions, description=f"Participant {participant.participant_id}"):
                session_dir = participant_dir / session_info.session_id
                if not session_dir.exists():
                    session_dir.mkdir(parents=True, exist_ok=True)
                
                # Migrate DICOM data to sourcedata/ subdirectory
                dicom_dir = self.dicom_root / session_info.dicom_subdir
                if symlink:
                    copy_as_symlinks(dicom_dir, session_dir)
                else:
                    shutil.copytree(dicom_dir, session_dir, dirs_exist_ok=True)
                    
    def migrate_dicom_per_session(self, participant_id: str, session_id: str, symlink: bool = True):
        """
        Migrate DICOM data to BIDS sourcedata/ subdirectory for a single session.
        """
        participant: Participant = self.participants.filter(participant_ids=[participant_id], bids_session_ids=[session_id]).participants[0]
        participant_dir = self.bids_root / "sourcedata" / participant.subject_id
        session_info = participant.sessions[0]
        session_dir = participant_dir / session_info.session_id
        dicom_dir = self.dicom_root / session_info.dicom_subdir
        if symlink:
            copy_as_symlinks(dicom_dir, session_dir)
        else:
            shutil.copytree(dicom_dir, session_dir, dirs_exist_ok=True)
            
    def run_dcm2bids_helper(self, subject_id: str, session_id: str, output_dir: PosixPath = Path("tmp/")) -> None:
        """
        Run dcm2bids_helper to create example sidecar json files.
        """
        participant_mapping = next((mapping for mapping in self.participants if mapping.subject_id == subject_id and mapping.session_id == session_id), None)
        dicom_subdir_full_path: PosixPath = self.dicom_root / participant_mapping.dicom_subdir
        subprocess.run(["dcm2bids_helper", "-d", str(dicom_subdir_full_path), "-o", str(output_dir)], check=True)

    def get_dcm2bids_cmd_per_participant(self, participant_id: str, bids_session_id: str) -> list[str]:
        """
        Get the dcm2bids command for a single participant.
        """
        participant: Participant = self.participants.filter(participant_ids=[participant_id], bids_session_ids=[bids_session_id]).participants[0]
        dicom_subdir: str = participant.sessions[0].dicom_subdir
        cmd: list[str] = ["dcm2bids", "-d", dicom_subdir, "-p", participant_id, "-s", bids_session_id, "-c", str(self.config), "-o", str(self.bids_root), "--auto_extract_entities"]
        return cmd
    
    def convert2bids_per_participant(self, participant_id: str, bids_session_id: str) -> None:
        """
        Convert DICOM data to BIDS format for a single participant.
        """
        subprocess.run(self.get_dcm2bids_cmd_per_participant(participant_id, bids_session_id), check=True)
        
    def convert2bids(self) -> None:
        """
        Convert DICOM data to BIDS format for all participants.
        """
        with multiprocessing.Pool() as pool:
            pool.starmap(self.convert2bids_per_participant, [(participant.participant_id, session_info.bids_session_id) for participant in self.participants for session_info in participant.sessions])
