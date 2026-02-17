"""
Load DICOM data and create mappings on participant level.
"""
from ..helpers.data import flatten_no_compound_key

from pathlib import Path, PosixPath
from typing import Optional, Iterable, Iterator
from collections.abc import MutableMapping
from itertools import islice
import csv, re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum

from pydantic import BaseModel, field_validator, ConfigDict
import pydicom as dicom
from rich.progress import track, Progress, TextColumn, BarColumn, TimeRemainingColumn
import pandas as pd
import asyncio

CAMEL_TO_SNAKE_CASE_PATTERN: re.Pattern = re.compile(r'(?<!^)(?=[A-Z])')
SNAKE_TO_CAMEL_CASE_PATTERN: re.Pattern = re.compile(r'(_)([a-z])')
PASCAL_TO_SNAKE_CASE_PATTERN: re.Pattern = re.compile(r'(?<!^)(?=[A-Z])')
SNAKE_TO_PASCAL_CASE_PATTERN: re.Pattern = re.compile(r'(_)([a-z])')

class StudyField(BaseModel):
    """
    Information about a study field.
    """
    variable_name: str
    tag_name: str
    value: Optional[str] = None

class Study(BaseModel):
    """
    Information about a study.
    """
    study_fields: list[StudyField] = []
    study_subdir: str
    study_name: str
    
    def add_study_field(self, variable_name: str, tag_name: str, value: Optional[str] = None) -> None:
        """
        Add a study field to the Study object.
        """
        if hasattr(self, variable_name) and getattr(self, variable_name) is not None:
            raise ValueError(f"Variable {variable_name} already exists in Study object")
        study_field: StudyField = StudyField(variable_name=variable_name, tag_name=tag_name, value=value)
        self.study_fields.append(study_field)
        
    def get_study_field(self, variable_name: str) -> Optional[str]:
        """
        Get the value of a study field by variable name.
        """
        for study_field in self.study_fields:
            if study_field.variable_name == variable_name:
                return study_field.value
        # If not found, load from DICOM files in the study_subdir
        if self.study_subdir:
            study_dir: PosixPath = Path(self.study_subdir)
            if study_dir.is_dir():
                dicom_files_iter = islice(study_dir.glob("*.dcm"), 5)
                dicom_files = list(dicom_files_iter)
                for dicom_file in dicom_files:
                    try:
                        ds = dicom.dcmread(dicom_file)
                        tag_name: str = snake_to_camel_case(variable_name)
                        value: Optional[str] = getattr(ds, tag_name, None)
                        if value is not None:
                            return str(value)
                    except Exception as e:
                        print(f"Error reading {dicom_file}: {e}")
                        continue
        return None
        
    @classmethod
    def from_xnat_session_subdir(cls, session_subdir: PosixPath) -> list["Study"]:
        """
        Create a list of Study objects from a session subdirectory in an XNAT-like structure.
        Assumes the following structure:
        <session_subdir>
        ├── <study_1>
        │   ├── DICOM
        │   │   ├── <dicom_files>
        """
        studies: list[Study] = []
        for study_dir in session_subdir.iterdir():
            if not study_dir.is_dir():
                continue
            dicom_dir: PosixPath = study_dir / "DICOM"
            if not dicom_dir.is_dir():
                continue
            study: Optional[Study] = determine_study_info(dicom_dir)
            if study:
                studies.append(study)
        return studies

class Session(BaseModel):
    """
    Session information for a participant.
    """
    session_id: str
    bids_session_id: Optional[str] = None
    dicom_subdir: str
    studies: list[Study]

    @field_validator("bids_session_id", mode="before")
    def convert_bids_session_id(cls, value):
        if isinstance(value, int):
            return f"{value:02d}"
        return value
    
    @classmethod
    def from_xnat_subject_subdir(cls, subject_subdir: PosixPath) -> list["Session"]:
        """
        Create a list of Session objects from a subject subdirectory in an XNAT-like structure.
        Assumes the following structure:
        <subject_subdir>
        ├── <session_id_1>
        │  ├── <study_1>
        │  │   ├── DICOM
        │  │   |   ├── <dicom_files>
        """
        sessions: list[Session] = []
        for bids_session_id, session_dir in enumerate(subject_subdir.iterdir(), start=1):
            if not session_dir.is_dir():
                continue
            session_id: str = session_dir.name
            # studies: list[Study] = []
            studies: list[Study] = Study.from_xnat_session_subdir(session_dir)
            if not studies:
                print(f"No studies found in session directory {session_dir}")
                continue
            sessions.append(Session(
                session_id=session_id,
                bids_session_id=bids_session_id,
                dicom_subdir=str(session_dir),
                studies=studies
            ))
        return sessions
    
    def add_study_field(self, variable_name: str, tag_name: str, value: Optional[str] = None) -> None:
        """
        Add a study field to all studies in the session.
        """
        for study in self.studies:
            study.add_study_field(variable_name, tag_name, value)

    def load_study_field(self, variable_name: str, append: bool = True) -> list[Optional[str]]:
        """
        Load the value of a study field from all studies in the session. If append is True, append the values to existing ones.
        """
        values: list[Optional[str]] = []
        for study in self.studies:
            value: Optional[str] = study.get_study_field(variable_name)
            values.append(value)
            if append and value is not None:
                study.add_study_field(variable_name, tag_name=snake_to_camel_case(variable_name), value=value)
        return values

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
            for study_info in sessions.studies:
                output += f"    {study_info.series_description}\n"
        return output
    
    def add_study_field(self, variable_name: str, tag_name: str, value: Optional[str] = None) -> None:
        """
        Add a study field to all studies in all sessions of the participant.
        """
        for session in self.sessions:
            session.add_study_field(variable_name, tag_name, value)
            
    def load_study_field(self, variable_name: str, append: bool = True) -> list[Optional[str]]:
        """
        Load the value of a study field from all studies in all sessions of the participant. If append is True, append the values to existing ones.
        """
        values: list[Optional[str]] = []
        for session in self.sessions:
            session_values: list[Optional[str]] = session.load_study_field(variable_name, append=append)
            values.extend(session_values)
        return values
    
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
    
    def __add__(self, other: Optional["Participants"]) -> "Participants":
        """
        Allow addition of two Participants objects.
        """
        if other is None:
            return self
        if not isinstance(other, Participants):
            raise TypeError("Can only add Participants objects")
        return Participants(participants=self.participants + other.participants)
    
    def __sub__(self, other: Optional["Participants"]) -> "Participants":
        """
        Allow subtraction of two Participants objects.
        """
        if other is None:
            return self
        if not isinstance(other, Participants):
            raise TypeError("Can only subtract Participants objects")
        other_ids: set[str] = set([p.participant_id for p in other.participants])
        filtered_participants: list[Participant] = [p for p in self.participants if p.participant_id not in other_ids]
        return Participants(participants=filtered_participants)
    
    @property
    def participant_ids(self) -> list[str]:
        """
        Get a list of participant IDs.
        """
        return [participant.participant_id for participant in self.participants]
    
    def pop(self, participant_id: str) -> Optional[Participant]:
        """
        Remove and return a participant by participant ID.
        """
        for idx, participant in enumerate(self.participants):
            if participant.participant_id == participant_id:
                return self.participants.pop(idx)
        return None
    
    @classmethod
    def from_table(cls, df: pd.DataFrame) -> "Participants":
        """
        Create a Participants object from a list of dictionaries.
        """
        # Convert all columns to strings
        df = df.astype(str)
        all_columns: list[str] = df.columns.tolist()
        session_cols: list[str] = [col for col in all_columns if col not in ["subject_id", "participant_id"]]
        # study_cols: list[str] = [col for col in session_cols if col not in ["session_id", "bids_session_id", "dicom_subdir", "study_subdir"]]
        study_cols: list[str] = [col for col in session_cols if col not in ["session_id", "bids_session_id", "dicom_subdir"]]

        participant_ids: list[str] = df["participant_id"].unique().tolist()
        participants_dict: list[Participant] = []
        for participant_id in track(participant_ids, description="Loading participants from table"):
            subject_id: str = df[df["participant_id"] == participant_id]["subject_id"].iloc[0]
            # Get all session info for the current participant
            session_rows: list[dict] = df[df["participant_id"] == participant_id][session_cols].to_dict(orient="records")
            sessions_dict: list[Session] = []
            for session_row in session_rows:
                session_id: str = session_row["session_id"] if "session_id" in session_row else None
                bids_session_id: str = session_row["bids_session_id"] if "bids_session_id" in session_row else None
                dicom_subdir: str = session_row["dicom_subdir"] if "dicom_subdir" in session_row else None
                # Get all study info for the current session
                study_rows: list[dict] = df[(df["participant_id"] == participant_id) & (df["session_id"] == session_id)][study_cols].to_dict(orient="records")
                studies_dict: list[Study] = []
                for study_row in study_rows:
                    study_fields: list[StudyField] = []
                    for col in study_cols:
                        if col in study_row and study_row[col] is not None:
                            variable_name: str = camel_to_snake_case(col)
                            study_fields.append(StudyField(variable_name=variable_name, tag_name=col, value=study_row[col]))
                    study: Study = Study(study_fields=study_fields, study_subdir=study_row["study_subdir"] if "study_subdir" in study_row else None, study_name=study_row["study_name"] if "study_name" in study_row else "Unknown")
                    studies_dict.append(study)
                sessions_dict.append(Session(
                    session_id=session_id,
                    bids_session_id=bids_session_id,
                    dicom_subdir=dicom_subdir,
                    studies=studies_dict
                ))
            participant_dict: Participant = Participant(
                subject_id=subject_id,
                participant_id=participant_id,
                sessions=sessions_dict
            )
            participants_dict.append(participant_dict)
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
        
    def add_study_field(self, variable_name: str, tag_name: str, value: Optional[str] = None) -> None:
        """
        Add a study field to all studies in all sessions of all participants.
        """
        for participant in self.participants:
            participant.add_study_field(variable_name, tag_name, value)
    
    # ! Need further testing
    def load_study_field(self, variable_name: str, append: bool = True) -> list[Optional[str]]:
        """
        Load the value of a study field from all studies in all sessions of all participants. If append is True, append the values to existing ones.
        """
        values: list[Optional[str]] = []
        for participant in self.participants:
            participant_values: list[Optional[str]] = participant.load_study_field(variable_name, append=append)
            values.extend(participant_values)
        return values
    
    def to_table(self, skip_columns: Optional[Iterable[str]] = None, to_df: bool = False) -> list[dict] | pd.DataFrame:
        """
        Convert participant mappings to a list of flattened dictionaries.
        """
        rows: list[dict] = []
        for participant in self.participants:
            for session in participant.sessions:
                for study in session.studies:
                    row: dict = {
                        "subject_id": participant.subject_id,
                        "participant_id": participant.participant_id,
                        "session_id": session.session_id,
                        "bids_session_id": session.bids_session_id,
                        "dicom_subdir": session.dicom_subdir,
                        "study_subdir": study.study_subdir,
                        "study_name": study.study_name,
                    }

                    for study_field in study.study_fields:
                        row[study_field.variable_name] = study_field.value

                    if skip_columns:
                        row = {k: v for k, v in row.items() if k not in skip_columns}
                    rows.append(row)
                    
        if to_df:
            return pd.DataFrame(rows)
        return rows

# ============================DICOM Loaders============================

def camel_to_snake_case(name: str) -> str:
    """
    Convert a camelCase string to snake_case.
    """
    return CAMEL_TO_SNAKE_CASE_PATTERN.sub("_", name).lower()

def snake_to_camel_case(name: str) -> str:
    """
    Convert a snake_case string to camelCase.
    """
    return SNAKE_TO_CAMEL_CASE_PATTERN.sub(lambda x: x.group(2).upper(), name).replace("_", "")

def pascal_to_snake_case(name: str) -> str:
    """
    Convert a PascalCase string to snake_case.
    """
    return PASCAL_TO_SNAKE_CASE_PATTERN.sub("_", name).lower()

def snake_to_pascal_case(name: str) -> str:
    """
    Convert a snake_case string to PascalCase.
    """
    # First capitalize each word after underscore using the pattern
    result = SNAKE_TO_PASCAL_CASE_PATTERN.sub(lambda x: x.group(2).upper(), name)
    # Then capitalize the first letter and remove all underscores
    return result[0].upper() + result[1:].replace("_", "")

def determine_study_info(study_dir: Path, study_name: str, max_files: int = 5, force_read: bool = False) -> Optional[Study]:
    """
    Determine study information from DICOM files in a study directory.
    Reads up to `max_files` DICOM files in the directory to extract study-level metadata
    Args:
        study_dir (Path): Path to the study directory containing DICOM files.
        max_files (int): Maximum number of DICOM files to read for extracting metadata.
        force_read (bool): Whether to force reading of DICOM files even if they are missing headers.
    Returns:
        Study: Study object containing extracted metadata, or None if no valid DICOM files are found.
    """
    if not study_dir.is_dir():
        raise ValueError(f"{study_dir} is not a directory")

    # DICOM files should only be in the study directory, not in subdirectories - so no recursive glob
    dicom_files_iter = islice(study_dir.glob("*.dcm"), max_files)
    dicom_files = list(dicom_files_iter)
    if not dicom_files:
        print(f"No DICOM files found in {study_dir}")
        return None

    for idx, dicom_file in enumerate(dicom_files):
        try:
            ds = dicom.dcmread(dicom_file, force=force_read)
            
            study_tag_names: list = [
                "StudyDate",
                "AcquisitionTime",
                "SeriesDescription",
                "SeriesNumber",
                "ProtocolName",
                "Manufacturer",
                "ManufacturerModelName",
                "MagneticFieldStrength",
                "RepetitionTime",
                "EchoTime",
                "SliceThickness",
                "PixelSpacing",
                "ContrastBolusAgent",
                "FlipAngle",
                "Rows",
                "Columns",
                "NumberOfFrames",
                "ImagePositionPatient",
                "ImageOrientationPatient",
                "TemporalPositionIdentifier",
                "NumberOfTemporalPositions",
            ]

            study_fields: list[StudyField] = []

            for tag_name in study_tag_names:
                value: Optional[str] = getattr(ds, tag_name, None)
                if value is not None:
                    value = str(value)
                variable_name: str = camel_to_snake_case(tag_name)
                study_field: StudyField = StudyField(variable_name=variable_name, tag_name=tag_name, value=value)
                if hasattr(Study, variable_name):
                    raise ValueError(f"Variable {variable_name} already exists in Study object")
                study_fields.append(study_field)
                
            return Study(study_fields=study_fields, study_subdir=str(study_dir), study_name=study_name)
            
        except Exception as e:
            print(f"Error reading {dicom_file}: {e}")
            continue

    print(f"No valid DICOM files found in {study_dir} after checking {min(max_files, len(dicom_files))} files.")
    return None
        


def load_dicom_structured_sessions(dicom_root: PosixPath, sample: Optional[int] = None) -> Participants:
    """
    DICOM Loader if the DICOM dataset has the following features-
    1. The directory structure is organized by session ID and studies.
    2. Study subdirectories contain DICOM files.

    Example tree-
    <root>
    ├── <session_id_1>
    │   ├── <study_1>
    │   │   ├── <dicom_files>
    """
    participant_mappings: list[Participant] = []
    
    session_ids: list[str] = sorted([subdir.name for subdir in dicom_root.iterdir() if subdir.is_dir()])
    subject_ids: list[str] = sorted(set(["-".join(session_id.split("-")[:2]) for session_id in session_ids]))    
    
    end: int = sample if sample else len(subject_ids)
    for participant_id, subject_id in enumerate(subject_ids[:end], start=1):
        print(f"Loading participant {participant_id} ({subject_id})")
        session_subdirs: list[PosixPath] = sorted([subdir for subdir in dicom_root.iterdir() if subdir.is_dir() and subdir.name.startswith(subject_id)])
        session_info: list[Session] = []
        for bids_session_id, session_dir in enumerate(session_subdirs, start=1):
            if not session_dir.is_dir():
                continue
            session_id: str = session_dir.name
            # dicom_subdir: str = session.name
            studies: list[Study] = []
            for study_dir in session_dir.iterdir():
                if not study_dir.is_dir():
                    continue
                try:
                    study: Optional[Study] = determine_study_info(study_dir)
                except AttributeError as e:
                    print(e)
                    continue
                if study:
                    studies.append(study)
                
            session_info.append(Session(
                session_id=session_id,
                bids_session_id=bids_session_id,
                dicom_subdir=str(session_dir),
                studies=studies
                ))
            
        participant_mappings.append(
            Participant(
                subject_id=subject_id,
                participant_id=participant_id,
                sessions=session_info,
            )
        )
        
    return Participants(participants=participant_mappings)

def load_dicom_structured_subjects(dicom_root: PosixPath, sample: Optional[int] = None) -> Participants:
    """
    DICOM Loader if the DICOM dataset has the following features-
    1. The directory structure is organized by subject ID, session ID, and studies.
    2. Study subdirectories contain DICOM files.
    3. Subdirectory names are used as subject IDs and session IDs.
    
    Example tree-
    <root>
    ├── <subject_id_1>
    │   ├── <session_id_1>
    │   │   ├── <study_1>
    │   │   │   ├── <dicom_files>
    """
    participant_mappings: list[Participant] = []
    subject_id_pattern: re.Pattern = re.compile(r'^\d{4}-\d{4}$')  # Pattern to match subject IDs like '1234-5678'
    subject_ids: list[str] = sorted([subdir.name for subdir in dicom_root.iterdir() if subdir.is_dir() and subject_id_pattern.match(subdir.name)])
    session_id_pattern: re.Pattern = re.compile(r'^\d{4}-\d{4}-\d{4}-\d{4}$')  # Pattern to match session IDs like '1234-5678-9012-3456'

    end: int = sample if sample else len(subject_ids)
    
    print(f"Found {len(subject_ids)} subjects, loading {end} subjects...")

    for participant_id, subject_id in track(enumerate(subject_ids[:end], start=1), total=end, description="Loading participants"):
        session_subdirs: list[PosixPath] = sorted([subdir for subdir in (dicom_root / subject_id).iterdir() if subdir.is_dir() and session_id_pattern.match(subdir.name)])
        # print(f"Loading participant {participant_id} ({subject_id}) - number of sessions: {len(session_subdirs)}")
        session_info: list[Session] = []
        for bids_session_id, session in enumerate(session_subdirs, start=1):
            if not session.is_dir():
                continue
            session_id: str = session.name
            studies: list[Study] = []
            for study_dir in session.iterdir():
                if not study_dir.is_dir():
                    continue
                
                # Load study date from first DICOM file within the study directory, skip if directory is empty
                try:
                    study: Optional[Study] = determine_study_info(study_dir)
                except AttributeError as e:
                    print(f"Error loading {study_dir}: {e}")
                    continue
                if study:
                    studies.append(study)
                else:
                    print(f"No DICOM files found in {study_dir}")
                
            session_info.append(Session(
                session_id=session_id,
                bids_session_id=bids_session_id,
                dicom_subdir=str(session),
                studies=studies
                ))
            
        participant_mappings.append(
            Participant(
                subject_id=subject_id,
                participant_id=participant_id,
                sessions=session_info,
            )
        )
        
    return Participants(participants=participant_mappings)

def get_xnat_participant(subject_subdir: PosixPath, participant_id: int, temp_dir: Path) -> Participant:
    """
    Extract participant info and write to individual temp CSV file.
    """
    temp_dir: PosixPath = Path(temp_dir)
    if not subject_subdir.is_dir():
        print(f"Skipping missing subject dir: {subject_subdir}")
        return None
    out_csv = temp_dir / f"subject_{subject_subdir.name}.csv"
    if out_csv.exists():
        print(f"Participant {participant_id} already exists in {out_csv}, skipping.")
        return None

    sessions = Session.from_xnat_subject_subdir(subject_subdir)
    if not sessions:
        print(f"No sessions found in: {subject_subdir}")
        return None

    participant = Participant(
        subject_id=subject_subdir.name,
        participant_id=participant_id,
        sessions=sessions
    )

    # Save to temp CSV
    rows = Participants(participants=[participant]).to_table(to_df=False)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote participant {participant_id} from subject {subject_subdir.name} to {out_csv}")
    return participant

def load_xnat_dicom_subjects(dicom_root: PosixPath, csv_path: PosixPath, sample: Optional[int] = None, append_to_csv: bool = True) -> Participants:
    """
    DICOM Loader if the DICOM dataset has the following features-
    1. The directory structure is organized by subject ID, session ID, and studies.
    2. Study subdirectories contain DICOM files.
    3. Subdirectory names are used as subject IDs and session IDs.
    
    Example tree-
    <root>
    ├── <subject_id_1>
    │   ├── <session_id_1>
    │   │   ├── <study_1>
    │   │   │   ├── DICOM
    │   │   │   |   ├── <dicom_files>
    
    Process:
    - List all subject directories in the root directory and assign participant IDs.
    - Return each participant with their subject ID and sessions asynchronously.
    - If `csv_path` path exists, it will load existing participants from the CSV file. If it new participants will be added to a new CSV file of the same path.
    - If `sample` is provided, it will limit the number of subjects loaded to the specified sample size.
    - If `append_to_csv` is True, new participants will be appended to the CSV file specified by `csv_path`.
    
    Args:
        dicom_root (PosixPath): The root directory containing the DICOM files.
        csv_path (str): Path to a CSV file to resume from existing participant mappings. If the path exists, it will load existing participants from the CSV file. If it does not exist, it will create a new CSV file.
        sample (Optional[int]): Number of subjects to sample from the dataset. If None, all subjects are loaded.
        append_to_csv (bool): If True, new participants will be appended to the CSV file specified by `csv_path`. If False, nothing will be appended to the CSV file.
    """
    MAX_WORKERS: int = 8
    csv_temp_dir: Path = csv_path.parent / f"{csv_path.stem}_temp"
    csv_temp_dir.mkdir(exist_ok=True, parents=True)

    # Load existing IDs from final CSV
    existing_ids = set()
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        existing_ids = set(df_existing["subject_id"].unique())
        print(f"Found {len(existing_ids)} existing participants in {csv_path}")

    subject_dirs = [d for d in sorted(dicom_root.iterdir()) if d.is_dir() and d.name not in existing_ids]
    if sample:
        subject_dirs = subject_dirs[:sample]

    start_id = 1
    if existing_ids:
        start_id = 1 + len(existing_ids)

    # Parallel processing
    futures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for pid, subdir in enumerate(subject_dirs, start=start_id):
            futures.append(executor.submit(get_xnat_participant, subdir, pid, csv_temp_dir))

        for _ in track(as_completed(futures), total=len(futures), description="Loading participants"):
            _.result()  # discard result, already saved

    # Merge all individual CSVs
    temp_csvs = sorted(csv_temp_dir.glob("subject_*.csv"))
    print(f"Merging {len(temp_csvs)} temporary CSV files from {csv_temp_dir}")
    all_df = pd.concat([pd.read_csv(f) for f in temp_csvs], ignore_index=True) if temp_csvs else pd.DataFrame()

    if csv_path.exists():
        print(f"Appending new participants to existing CSV at {csv_path}")
        df_existing = pd.read_csv(csv_path)
        final_df = pd.concat([df_existing, all_df], ignore_index=True)
    else:
        final_df = all_df
        final_df.to_csv(csv_path, index=False)
        print(f"Merged {len(temp_csvs)} participants into {csv_path}")
        return Participants.from_table(final_df)


    # Optionally clean up temp CSVs
    # for f in temp_csvs:
    #     f.unlink()

    # csv_temp_dir.rmdir()
    
# def load_dicom_subjects_nested_sessions(dicom_root: PosixPath, csv_path: PosixPath, sample: Optional[int] = None, append_to_csv: bool = True) -> Participants:
#     """
#     DICOM Loader if the DICOM dataset has the following features-
#     1. The directory structure is organized by subject ID, session ID, and studies.
#     2. Study subdirectories contain DICOM files.
#     3. Subdirectory names are used as subject IDs and session IDs.
    
#     Example tree-
#     <root>
#     ├── <subject_id_1>
#     │   ├── <session_id_1>
#     │   │   ├── scans
#     │   │   │   ├── <study_1>
#     │   │   │   │   ├── resources
#     │   │   │   │   │   ├── DICOM
#     │   │   │   │   │   │   <dicom_files>
    
#     Args:
#         dicom_root (PosixPath): The root directory containing the DICOM files.
#         csv_path (str): Path to a CSV file to resume from existing participant mappings. If the path exists, it will load existing participants from the CSV file. If it does not exist, it will create a new CSV file.
#         sample (Optional[int]): Number of subjects to sample from the dataset. If None, all subjects are loaded.
#         append_to_csv (bool): If True, new participants will be appended to the CSV file specified by `csv_path`. If False, nothing will be appended to the CSV file.
#     """

def process_participant(
    participant_id: int,
    subject_id: str,
    dicom_root: Path,
    subject_subdir_pattern: Path,
    session_subdir_pattern: Path,
    study_subdir_pattern: Path,
    dicom_subdir_pattern: Path,
    force_read: bool,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
) -> Participant:
    """
    Process a single participant and return their Participant object.
    1. Navigate through the specified directory structure.
    2. Extract study information from DICOM files.
    3. Optionally update a CSV file with the participant's information.
    
    Args:
        participant_id (int): The participant ID.
        subject_id (str): The subject ID.
        dicom_root (Path): The root directory containing the DICOM files.
        subject_subdir_pattern (Path): Pattern for subject subdirectory.
        session_subdir_pattern (Path): Pattern for session subdirectory.
        study_subdir_pattern (Path): Pattern for study subdirectory.
        dicom_subdir_pattern (Path): Pattern for DICOM subdirectory.
        force_read (bool): If True, force reading of DICOM files even if they are missing headers.
    
    Returns:
        Participant: The processed participant object.
    """
    subject_dir = dicom_root / subject_subdir_pattern / subject_id
    session_subdirs = sorted([
        subdir for subdir in (subject_dir / session_subdir_pattern).iterdir()
        if subdir.is_dir()
    ])
    session_info: list[Session] = []

    if progress and task_id:
        progress.update(task_id, total=len(session_subdirs))

    for bids_session_id, session_dir in enumerate(session_subdirs, start=1):
        if progress and task_id:
            progress.update(task_id, description=f"{subject_id} → Session {bids_session_id}/{len(session_subdirs)}")

        if not session_dir.is_dir():
            continue

        session_id = session_dir.name
        studies: list[Study] = []

        for study_dir in (session_dir / study_subdir_pattern).iterdir():
            if not study_dir.is_dir():
                continue
            dicom_dir = study_dir / dicom_subdir_pattern
            if not dicom_dir.is_dir():
                continue
            try:
                study = determine_study_info(dicom_dir, study_name=study_dir.name, force_read=force_read)
            except AttributeError:
                continue
            if study:
                studies.append(study)

        session_info.append(Session(
            session_id=session_id,
            bids_session_id=bids_session_id,
            dicom_subdir=str(session_dir),
            studies=studies,
        ))

        if progress and task_id:
            progress.advance(task_id)

    participant = Participant(
        subject_id=subject_id,
        participant_id=participant_id,
        sessions=session_info,
    )

    if progress and task_id:
        progress.update(task_id, description=f"{subject_id} ✔ Done")

    return participant

def load_dicom_universal(
    dicom_root: Path | str,
    csv_path: Optional[Path | str] = None,
    subject_subdir_pattern: Path | str = "",
    session_subdir_pattern: Path | str = "",
    study_subdir_pattern: Path | str = "",
    dicom_subdir_pattern: Path | str = "",
    sample: Optional[int] = None,
    force_read: bool = False
    ) -> Participants:
    """
    Universal DICOM Loader that can handle various directory structures by specifying subdirectory patterns.
    Args:
        dicom_root (Path | str): The root directory containing the DICOM files.
        csv_path (Optional[Path | str]): Path to a CSV file to resume from existing participant mappings. If the path exists, it will load existing participants from the CSV file. If it does not exist, it will create a new CSV file.
        subject_subdir_pattern (Path | str): Pattern for subject subdirectory.
        session_subdir_pattern (Path | str): Pattern for session subdirectory.
        study_subdir_pattern (Path | str): Pattern for study subdirectory.
        dicom_subdir_pattern (Path | str): Pattern for DICOM subdirectory.
        sample (Optional[int]): Number of subjects to sample from the dataset. If None, all subjects are loaded.
        force_read (bool): If True, force reading of DICOM files even if they are missing headers.
    
    Returns:
        Participants: The loaded participant mappings.
    """
    dicom_root: Path = Path(dicom_root)
    csv_path: Path = Path(csv_path) if csv_path else None
    subject_subdir_pattern: Path = Path(subject_subdir_pattern)
    session_subdir_pattern: Path = Path(session_subdir_pattern)
    study_subdir_pattern: Path = Path(study_subdir_pattern)
    dicom_subdir_pattern: Path = Path(dicom_subdir_pattern)
    
    subject_ids: list[str] = sorted([subdir.name for subdir in (dicom_root / subject_subdir_pattern).iterdir() if subdir.is_dir()])
    
    # Load subject_ids from existing CSV if provided
    if csv_path and csv_path.exists():
        existing_participants: Participants = Participants.from_table(pd.read_csv(csv_path))
        existing_subject_ids = set([p.subject_id for p in existing_participants.participants])
        subject_ids = [sid for sid in subject_ids if sid not in existing_subject_ids]
        print(f"Resuming from {csv_path}, skipping {len(existing_subject_ids)} existing subjects.")
    
    end: int = sample if sample else len(subject_ids)
    participant_mappings: list[Participant] = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeRemainingColumn(),
    ) as progress:
        participants_task = progress.add_task("Loading participants", total=end)
        for participant_id, subject_id in enumerate(subject_ids[:end], start=1):
            participant = process_participant(
                participant_id=participant_id,
                subject_id=subject_id,
                dicom_root=dicom_root,
                subject_subdir_pattern=subject_subdir_pattern,
                session_subdir_pattern=session_subdir_pattern,
                study_subdir_pattern=study_subdir_pattern,
                dicom_subdir_pattern=dicom_subdir_pattern,
                force_read=force_read,
                csv_path=csv_path,
            )
            participant_mappings.append(participant)
            progress.update(participants_task, advance=1)
            
            # Append to CSV after each participant if csv_path is provided and exists
            if csv_path:
                if csv_path.exists():
                    df_participant: pd.DataFrame = Participants(participants=[participant]).to_table(to_df=True)
                    df_participant.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    Participants(participants=[participant]).to_table(to_df=True).to_csv(csv_path, index=False)

    return Participants(participants=participant_mappings)

async def async_process_participant(
    participant_id: int,
    subject_id: str,
    dicom_root: Path,
    subject_subdir_pattern: Path,
    session_subdir_pattern: Path,
    study_subdir_pattern: Path,
    dicom_subdir_pattern: Path,
    force_read: bool,
) -> Participant:
    """
    Async wrapper for process_participant to run CPU-bound work in executor.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        process_participant,
        participant_id,
        subject_id,
        dicom_root,
        subject_subdir_pattern,
        session_subdir_pattern,
        study_subdir_pattern,
        dicom_subdir_pattern,
        force_read,
        None,  # progress
        None,  # task_id
    )

async def async_save_participant_to_csv(participant: Participant, csv_path: Path) -> None:
    """
    Async save participant to CSV file.
    """
    loop = asyncio.get_event_loop()
    
    def _save_to_csv():
        df_participant = Participants(participants=[participant]).to_table(to_df=True)
        if csv_path.exists():
            df_participant.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df_participant.to_csv(csv_path, index=False)
    
    await loop.run_in_executor(None, _save_to_csv)

async def async_load_dicom_universal(
    dicom_root: Path | str,
    csv_path: Optional[Path | str] = None,
    subject_subdir_pattern: Path | str = "",
    session_subdir_pattern: Path | str = "",
    study_subdir_pattern: Path | str = "",
    dicom_subdir_pattern: Path | str = "",
    sample: Optional[int] = None,
    force_read: bool = False,
    max_concurrent: int = 4
) -> Participants:
    """
    Async version of universal DICOM Loader that processes participants concurrently.
    
    Args:
        dicom_root (Path | str): The root directory containing the DICOM files.
        csv_path (Optional[Path | str]): Path to a CSV file to resume from existing participant mappings.
        subject_subdir_pattern (Path | str): Pattern for subject subdirectory.
        session_subdir_pattern (Path | str): Pattern for session subdirectory.
        study_subdir_pattern (Path | str): Pattern for study subdirectory.
        dicom_subdir_pattern (Path | str): Pattern for DICOM subdirectory.
        sample (Optional[int]): Number of subjects to sample from the dataset.
        force_read (bool): If True, force reading of DICOM files even if they are missing headers.
        max_concurrent (int): Maximum number of concurrent participants to process.
    
    Returns:
        Participants: The loaded participant mappings.
    """
    dicom_root = Path(dicom_root)
    csv_path = Path(csv_path) if csv_path else None
    subject_subdir_pattern = Path(subject_subdir_pattern)
    session_subdir_pattern = Path(session_subdir_pattern)
    study_subdir_pattern = Path(study_subdir_pattern)
    dicom_subdir_pattern = Path(dicom_subdir_pattern)
    
    subject_ids = sorted([subdir.name for subdir in (dicom_root / subject_subdir_pattern).iterdir() if subdir.is_dir()])
    
    # Load subject_ids from existing CSV if provided
    existing_participants = None
    if csv_path and csv_path.exists():
        existing_participants = Participants.from_table(pd.read_csv(csv_path))
        existing_subject_ids = set([p.subject_id for p in existing_participants.participants])
        subject_ids = [sid for sid in subject_ids if sid not in existing_subject_ids]
        print(f"Resuming from {csv_path}, skipping {len(existing_subject_ids)} existing subjects.")
    
    end = sample if sample else len(subject_ids)
    participant_mappings = []

    # Process participants in batches for controlled concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_and_save_participant(participant_id: int, subject_id: str):
        async with semaphore:
            print(f"Processing participant {participant_id} ({subject_id})")
            participant = await async_process_participant(
                participant_id=participant_id,
                subject_id=subject_id,
                dicom_root=dicom_root,
                subject_subdir_pattern=subject_subdir_pattern,
                session_subdir_pattern=session_subdir_pattern,
                study_subdir_pattern=study_subdir_pattern,
                dicom_subdir_pattern=dicom_subdir_pattern,
                force_read=force_read,
            )
            
            # Save to CSV if path provided
            if csv_path:
                await async_save_participant_to_csv(participant, csv_path)
                print(f"Saved participant {participant_id} ({subject_id}) to {csv_path}")
            
            return participant

    # Create tasks for all participants
    tasks = [
        process_and_save_participant(participant_id, subject_id)
        for participant_id, subject_id in enumerate(subject_ids[:end], start=1)
    ]
    
    print(f"Starting async processing of {len(tasks)} participants with max_concurrent={max_concurrent}")
    
    # Execute all tasks and collect results
    participants = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and collect successful participants
    for result in participants:
        if isinstance(result, Exception):
            print(f"Error processing participant: {result}")
        else:
            participant_mappings.append(result)

    # Combine with existing participants if any
    all_participants = Participants(participants=participant_mappings)
    if existing_participants:
        all_participants = existing_participants + all_participants

    return all_participants

