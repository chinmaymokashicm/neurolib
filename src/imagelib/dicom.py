"""
Load DICOM data and create mappings on participant level.
"""

from pathlib import Path, PosixPath
from typing import Optional, Iterable

from pydantic import BaseModel, field_validator
import pydicom as dicom
from rich.progress import track

class ScanInfo(BaseModel):
    """
    Information about a scan.
    """
    scan_name: str
    scan_type: Optional[str] = None
    scan_date: Optional[str] = None
    series_number: Optional[str] = None
    acquisition_time: Optional[str] = None
    
    @field_validator("series_number", mode="before")
    def convert_series_number(cls, value):
        if value is not None:
            return str(value)
        return value

class SessionInfo(BaseModel):
    """
    Session information for a participant.
    """
    session_id: str
    bids_session_id: Optional[str] = None
    dicom_subdir: str
    scans: list[ScanInfo]
    
    @field_validator("bids_session_id", mode="before")
    def convert_bids_session_id(cls, value):
        if isinstance(value, int):
            return f"{value:02d}"
        return value

class ParticipantInfo(BaseModel):
    """
    Information about a participant, including their original subject ID and sessions.
    """
    subject_id: str
    participant_id: str
    session_info: list[SessionInfo]
    
    @field_validator("participant_id", mode="before")
    def convert_participant_id(cls, value):
        if isinstance(value, int):
            return f"{value:04d}"
        raise ValueError(f"Pass an integer value for participant_id, got {value} of type {type(value)}")
    
    def __str__(self):
        """
        Print the participant mappings in a human-readable format.
        """
        output: str = f"Participant {self.participant_id} ({self.subject_id})\n"
        for session_info in self.session_info:
            output += f"  {session_info.session_id}\n"
            for scan_info in session_info.scans:
                output += f"    {scan_info.scan_name}\n"
        return output
    
def convert_participant_mappings2table(participant_mappings: list[ParticipantInfo], skip_columns: Optional[Iterable[str]] = None) -> list[dict]:
    """
    Convert a list of participant mappings to a list of flattened dictionaries.
    """
    table = []
    for participant_mapping in participant_mappings:
        for session_info in participant_mapping.session_info:
            for scan_info in session_info.scans:
                row: dict = {
                    "subject_id": participant_mapping.subject_id,
                    "participant_id": participant_mapping.participant_id,
                    "session_id": session_info.session_id,
                    "bids_session_id": session_info.bids_session_id,
                    "dicom_subdir": session_info.dicom_subdir,
                    "scan_name": scan_info.scan_name,
                    "scan_type": scan_info.scan_type,
                    "scan_date": scan_info.scan_date,
                    "series_number": scan_info.series_number,
                    "acquisition_time": scan_info.acquisition_time,
                }
                if skip_columns:
                    row = {k: v for k, v in row.items() if k not in skip_columns}
                table.append(row)
    return table

# ============================DICOM Loaders============================

def load_dicom_structured_sessions(dicom_root: PosixPath, sample: Optional[int] = None) -> list[ParticipantInfo]:
    """
    DICOM Loader if the DICOM dataset has the following features-
    1. The directory structure is organized by subject ID, session ID, and scans.
    2. Scan subdirectories contain DICOM files.
    3. Subdirectory names are used as subject IDs and session IDs.
    """
    participant_mappings: list[ParticipantInfo] = []
    
    session_ids: list[str] = [subdir.name for subdir in dicom_root.iterdir() if subdir.is_dir()]
    subject_ids: list[str] = sorted(list(set(["-".join(session_id.split("-")[:2]) for session_id in session_ids])))
    
    end: int = sample if sample else len(subject_ids)
    for participant_id, subject_id in enumerate(subject_ids[:end], start=1):
        print(f"Processing participant {participant_id} ({subject_id})")
        session_subdirs: list[PosixPath] = sorted([subdir for subdir in dicom_root.iterdir() if subdir.is_dir() and subdir.name.startswith(subject_id)])
        session_info: list[SessionInfo] = []
        for bids_session_id, session in enumerate(session_subdirs, start=1):
            if not session.is_dir():
                continue
            session_id: str = session.name
            dicom_subdir: str = session.name
            scans: list[ScanInfo] = []
            for scan in session.iterdir():
                if not scan.is_dir():
                    continue
                
                # Load scan date from first DICOM file within the scan directory, skip if directory is empty
                first_dicom_file: PosixPath = next(scan.glob("*.dcm"), None)
                if first_dicom_file:
                    scan_date = dicom.dcmread(first_dicom_file).StudyDate
                    if not scan_date:
                        scan_date = dicom.dcmread(first_dicom_file).AcquisitionDate
                    if not scan_date:
                        print(f"No date found for {first_dicom_file}")
                    scan_name: str = dicom.dcmread(first_dicom_file).SeriesDescription
                    series_number = dicom.dcmread(first_dicom_file).SeriesNumber
                    acquisition_time = dicom.dcmread(first_dicom_file).AcquisitionTime.split(".")[0]
                    
                    # Determine scan type based on SeriesDescription, MRAcquisitionType, and DiffusionBValue
                    if any(keyword in scan_name.lower() for keyword in ["t1", "mprage", "t2"]):
                        scan_type = "anat"
                    elif any(keyword in scan_name.lower() for keyword in ["dwi", "dti"]) or "DiffusionBValue" in dicom.dcmread(first_dicom_file):
                        scan_type = "dwi"
                    elif "bold" in scan_name.lower():
                        scan_type = "func"
                    else:
                        scan_type = "unexpected"
                else:
                    print(f"No DICOM files found in {scan}")
                    
                scans.append(ScanInfo(
                    scan_name=scan_name, 
                    scan_type=scan_type, 
                    scan_date=scan_date, 
                    series_number=series_number,
                    acquisition_time=acquisition_time,
                    ))
                
            session_info.append(SessionInfo(
                session_id=session_id,
                bids_session_id=bids_session_id,
                dicom_subdir=dicom_subdir,
                scans=scans
                ))
            
        participant_mappings.append(
            ParticipantInfo(
                subject_id=subject_id,
                participant_id=participant_id,
                session_info=session_info,
            )
        )
        
    return participant_mappings