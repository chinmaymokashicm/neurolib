"""
Load DICOM data and create mappings on participant level.
"""
from .helpers.data import flatten_no_compound_key

from pathlib import Path, PosixPath
from typing import Optional, Iterable, Iterator
from collections.abc import MutableMapping

from pydantic import BaseModel, field_validator
import pydicom as dicom
from rich.progress import track
import pandas as pd

class Scan(BaseModel):
    """
    Information about a scan.
    """
    scan_name: str
    scan_type: Optional[str] = None
    scan_date: Optional[str] = None
    series_number: Optional[str] = None
    acquisition_time: Optional[str] = None
    scan_subdir: str
    
    @field_validator("series_number", mode="before")
    def convert_series_number(cls, value):
        if value is not None:
            return str(value)
        return value

class Session(BaseModel):
    """
    Session information for a participant.
    """
    session_id: str
    bids_session_id: Optional[str] = None
    dicom_subdir: str
    scans: list[Scan]
    
    @field_validator("bids_session_id", mode="before")
    def convert_bids_session_id(cls, value):
        if isinstance(value, int):
            return f"{value:02d}"
        return value

class Participant(BaseModel):
    """
    Information about a participant, including their original subject ID and sessions.
    """
    subject_id: str
    participant_id: str
    sessions: list[Session]
    
    @field_validator("participant_id", mode="before")
    def convert_participant_id(cls, value):
        if isinstance(value, int):
            return f"{value:04d}"
        return value
    
    def __str__(self):
        """
        Print the participant mappings in a human-readable format.
        """
        output: str = f"Participant {self.participant_id} ({self.subject_id})\n"
        for sessions in self.sessions:
            output += f"  {sessions.session_id}\n"
            for scan_info in sessions.scans:
                output += f"    {scan_info.scan_name}\n"
        return output
    
class Participants(BaseModel):
    participants: list[Participant]
    
    def __str__(self):
        """
        Print the participant mappings in a human-readable format.
        """
        output: str = ""
        for participant in self.participants:
            output += str(participant)
        return output
    
    def __iter__(self) -> Iterator[Participant]:
        """
        Allow iteration over participants.
        """
        return iter(self.participants)
    
    @classmethod
    def from_table(cls, df: pd.DataFrame) -> "Participants":
        """
        Create a Participants object from a list of dictionaries.
        """
        participant_ids: list[str] = df["participant_id"].unique().tolist()
        participants_dict: list[Participant] = []
        for participant_id in participant_ids:
            subject_id: str = df[df["participant_id"] == participant_id]["subject_id"].iloc[0]
            participant_dict: dict = {"participant_id": int(participant_id), "subject_id": subject_id, "sessions": []}
            bids_session_ids: list[str] = df[df["participant_id"] == participant_id]["bids_session_id"].unique().tolist()
            for bids_session_id in bids_session_ids:
                session_id, dicom_subdir, bids_session_id = df[(df["participant_id"] == participant_id) & (df["bids_session_id"] == bids_session_id)][["session_id", "dicom_subdir", "bids_session_id"]].iloc[0]
                scan_names: list[str] = df[(df["participant_id"] == participant_id) & (df["bids_session_id"] == bids_session_id)]["scan_name"].unique().tolist()
                session_dict: dict = {"session_id": session_id, "dicom_subdir": dicom_subdir, "bids_session_id": bids_session_id, "scans": []}
                for scan_name in scan_names:
                    scan_type, scan_date, series_number, acquisition_time, scan_subdir = df[(df["participant_id"] == participant_id) & (df["bids_session_id"] == bids_session_id) & (df["scan_name"] == scan_name)][["scan_type", "scan_date", "series_number", "acquisition_time", "scan_subdir"]].iloc[0]
                    scan_dict: dict = {"scan_name": scan_name, "scan_type": scan_type, "scan_date": scan_date, "series_number": series_number, "acquisition_time": acquisition_time, "scan_subdir": scan_subdir}
                    session_dict["scans"].append(Scan(**scan_dict))
                participant_dict["sessions"].append(Session(**session_dict))
            participants_dict.append(Participant(**participant_dict))
            
        participants: Participants = cls(participants=participants_dict)
        return participants
    
    def filter(self, participant_ids: Optional[list[str]] = None, bids_session_ids: Optional[list[str]] = None) -> "Participants":
        """
        Filter participants based on participant IDs and BIDS session IDs.
        """
        if not participant_ids and not bids_session_ids:
            return self
        filtered_participants: list[Participant] = []
        for participant in self.participants:
            if not participant.participant_id in participant_ids:
                continue
            filtered_sessions: list[Session] = participant.sessions
            if bids_session_ids:
                filtered_sessions: list[Session] = [session for session in participant.sessions if session.bids_session_id in bids_session_ids]
            filtered_participants.append(Participant(subject_id=participant.subject_id, participant_id=participant.participant_id, sessions=filtered_sessions))
        return Participants(participants=filtered_participants)
        
    
    def to_table(self, skip_columns: Optional[Iterable[str]] = None, to_df: bool = False) -> list[dict] | pd.DataFrame:
        """
        Convert participant mappings to a list of flattened dictionaries.
        """
        rows: list[dict] = []
        for participant in self.participants:
            for session in participant.sessions:
                for scan in session.scans:
                    row: dict = {
                        "subject_id": participant.subject_id,
                        "participant_id": participant.participant_id,
                        "session_id": session.session_id,
                        "bids_session_id": session.bids_session_id,
                        "dicom_subdir": session.dicom_subdir,
                        "scan_name": scan.scan_name,
                        "scan_type": scan.scan_type,
                        "scan_date": scan.scan_date,
                        "series_number": scan.series_number,
                        "acquisition_time": scan.acquisition_time,
                        "scan_subdir": scan.scan_subdir
                    }
                    if skip_columns:
                        row = {k: v for k, v in row.items() if k not in skip_columns}
                    rows.append(row)
                    
        if to_df:
            return pd.DataFrame(rows)
        return rows

# ============================DICOM Loaders============================

def load_dicom_structured_sessions(dicom_root: PosixPath, sample: Optional[int] = None) -> Participants:
    """
    DICOM Loader if the DICOM dataset has the following features-
    1. The directory structure is organized by subject ID, session ID, and scans.
    2. Scan subdirectories contain DICOM files.
    3. Subdirectory names are used as subject IDs and session IDs.
    """
    participant_mappings: list[Participant] = []
    
    session_ids: list[str] = sorted([subdir.name for subdir in dicom_root.iterdir() if subdir.is_dir()])
    subject_ids: list[str] = sorted(set(["-".join(session_id.split("-")[:2]) for session_id in session_ids]))    
    
    end: int = sample if sample else len(subject_ids)
    for participant_id, subject_id in enumerate(subject_ids[:end], start=1):
        print(f"Loading participant {participant_id} ({subject_id})")
        session_subdirs: list[PosixPath] = sorted([subdir for subdir in dicom_root.iterdir() if subdir.is_dir() and subdir.name.startswith(subject_id)])
        session_info: list[Session] = []
        for bids_session_id, session in enumerate(session_subdirs, start=1):
            if not session.is_dir():
                continue
            session_id: str = session.name
            # dicom_subdir: str = session.name
            scans: list[Scan] = []
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
                    elif any(keyword in scan_name.lower() for keyword in ["bold", "task", "rest"]):
                        scan_type = "func"
                    else:
                        scan_type = "unexpected"
                else:
                    print(f"No DICOM files found in {scan}")
                    
                scans.append(Scan(
                    scan_name=scan_name, 
                    scan_type=scan_type, 
                    scan_date=scan_date, 
                    series_number=series_number,
                    acquisition_time=acquisition_time,
                    scan_subdir=str(scan)
                    ))
                
            session_info.append(Session(
                session_id=session_id,
                bids_session_id=bids_session_id,
                dicom_subdir=str(session),
                scans=scans
                ))
            
        participant_mappings.append(
            Participant(
                subject_id=subject_id,
                participant_id=participant_id,
                sessions=session_info,
            )
        )
        
    return Participants(participants=participant_mappings)