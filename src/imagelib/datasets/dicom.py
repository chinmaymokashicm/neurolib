"""
Subject -1:N-> Session -1:N-> Series -1:N-> Scan
"""
import re
from typing import Optional, Self, Any, Sequence
from pathlib import Path
from datetime import date
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from pydantic import BaseModel, Field
import pandas as pd
from rich.progress import track
import pydicom as dicom

DICOM_TAGS: list[str] = [
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
    "PatientPosition",
    "PatientSex",
    "PatientAge",
    "PatientSize",
    "PatientWeight",
    "BodyPartExamined",
    "ScanningSequence",
]

class MRNCrosswalkEntry(BaseModel):
    subject_id: str
    mrn: str
    study_date: Optional[date] = None
    
class MRNCrosswalk(BaseModel):
    entries: list[MRNCrosswalkEntry] = Field(default_factory=list)
    
    @classmethod
    def from_csv(cls, csv_path: Path | str, subject_colname: str = 'Subject', mrn_colname: str = 'PatientID', study_date_colname: Optional[str] = 'StudyDate') -> Self:
        df_crosswalk: pd.DataFrame = pd.read_csv(csv_path, dtype=str)
        df_mapping: pd.DataFrame = df_crosswalk[[subject_colname, mrn_colname, study_date_colname]].dropna()
        df_mapping[study_date_colname] = pd.to_datetime(
            df_mapping[study_date_colname],
            format='%Y%m%d',
            errors='coerce'
        ).dt.date
        if study_date_colname is not None:
            df_mapping = df_mapping.dropna(subset=[study_date_colname])
        entries = [
            MRNCrosswalkEntry(
                subject_id=row[subject_colname],
                mrn=row[mrn_colname],
                study_date=row[study_date_colname] if study_date_colname is not None else None)
            for _, row in df_mapping.iterrows()
        ]
        return cls(entries=entries)
    
    def get_mrn(self, subject_id: str) -> Optional[str]:
        for entry in self.entries:
            if entry.subject_id == subject_id:
                return entry.mrn
        return None
    
    def get_study_date(self, subject_id: str) -> Optional[date]:
        for entry in self.entries:
            if entry.subject_id == subject_id:
                return entry.study_date
        return None

class ScanField(BaseModel):
    variable_name: str
    dicom_tag: str
    description: Optional[str] = None
    value: Optional[Any] = None

# class Scan(BaseModel):
#     name: str
#     scan_fields: list[ScanField] = Field(default_factory=list)
    
#     @classmethod
#     def from_dicom_file(cls, dicom_path: Path | str, name: Optional[str] = None) -> Self:
#         dicom_path = Path(dicom_path)
#         dicom_data = dicom.dcmread(dicom_path)
#         scan_fields = [ScanField(variable_name=tag, dicom_tag=tag, value=getattr(dicom_data, tag, None)) for tag in DICOM_TAGS]
#         return cls(name=name if name is not None else dicom_path.stem, scan_fields=scan_fields)
    
class Series(BaseModel):
    name: str
    # scans: list[Scan] = Field(default_factory=list)
    n_scans: int
    series_description: Optional[str] = None
    series_date: Optional[date] = None
    
    @classmethod
    def from_dicom_series(cls, dicom_series: Sequence[Path | str], name: Optional[str] = None) -> Self:
        # Extract series-level metadata from the first DICOM file in the series, and count the number of scans in the series
        if len(dicom_series) == 0:
            raise ValueError("DICOM series must contain at least one DICOM file.")
        dicom_data = dicom.dcmread(dicom_series[0])
        series_description = getattr(dicom_data, "SeriesDescription", None)
        series_date = getattr(dicom_data, "StudyDate", None)
        if series_date is not None:
            series_date = pd.to_datetime(series_date, format='%Y%m%d', errors='coerce').date()
        n_scans = len(dicom_series)
        return cls(name=name if name is not None else Path(dicom_series[0]).parent.name, n_scans=n_scans, series_description=series_description, series_date=series_date)
    
class Session(BaseModel):
    name: str
    series: list[Series] = Field(default_factory=list)
    study_date: Optional[date] = None
    
class Subject(BaseModel):
    mrn: Optional[str] = None
    name: str
    sessions: list[Session] = Field(default_factory=list)
    
class Subjects(BaseModel):
    subjects: list[Subject] = Field(default_factory=list)
    
    def __getitem__(self, subject_name: str) -> Optional[Subject]:
        for subject in self.subjects:
            if subject.name == subject_name:
                return subject
        return None
    
    def __str__(self) -> str:
        output = []
        for subject in self.subjects:
            output.append(f"Subject: {subject.name} (MRN: {subject.mrn})")
            for session in subject.sessions:
                output.append(f"  Session: {session.name} (Date: {session.study_date})")
                for series in session.series:
                    output.append(f"    Series: {series.name} (Description: {series.series_description}) - {series.n_scans} scans)")
        return "\n".join(output)
    
    @classmethod
    def from_flat_sessions_dicom_dir(
        cls,
        dicom_root: Path | str,
        series_subdir_pattern: Path | str = "",
        dicom_subdir_pattern: Path | str = "",
        mrn_crosswalk_path: Optional[Path | str] = None,
        filter_by_mrn: bool = True,
        sample: Optional[int] = None,
        load_dicom_fields: bool = False
    ) -> Self:
        """
        Assumes a directory structure of:
        Session ID/
            Series ID/
                DICOM files...
        where Session ID is unique across all subjects.
        The pattern arguments append to the end of the respective subdirectory names, and can be used to filter for specific sessions/series/dicoms.
        
        Args:
            dicom_root: Root directory containing session subdirectories.
            series_subdir_pattern: Optional pattern to append to the end of series subdirectory names for filtering (e.g. "SCANS/").
            dicom_subdir_pattern: Optional pattern to append to the end of dicom subdirectory names for filtering (e.g. "DICOM/").
            mrn_crosswalk_path: Optional path to CSV file containing MRN crosswalk information.
            filter_by_mrn: Whether to filter subjects by MRN using the crosswalk. If True, only subjects with MRN information in the crosswalk will be included.
            sample: Optional integer to limit the number of sessions processed (for testing purposes).
            load_dicom_fields: Whether to load DICOM fields for each scan. If False, only the scan name will be loaded without the additional DICOM metadata.
        """
        dicom_root = Path(dicom_root)
        
        subjects: Self = cls()
        
        # Use the crosswalk to map subject ID with MRN and keep only those subjects going forward if filter_by_mrn is True
        mrn_crosswalk = None
        if mrn_crosswalk_path is not None:
            mrn_crosswalk = MRNCrosswalk.from_csv(mrn_crosswalk_path)
        if filter_by_mrn and mrn_crosswalk is None:
            raise ValueError("MRN crosswalk is required for filtering by MRN, but no mapping was found.")
        
        # Identify each subdirectory in the root as a session if it is in the pattern of XXXX-XXXX-XXXX-XXXX
        subjects_list: list[Subject] = []
        sessions_list: list[Session] = []
        sessions: list[tuple[str, Path]] = [(subdir.name, subdir) for subdir in dicom_root.iterdir() if subdir.is_dir() and re.match(r'^\d{4}-\d{4}-\d{4}-\d{4}$', subdir.name)]
        if sample is not None:
            sessions = sessions[:sample]
        for session_id, session_path in track(sessions, description="Processing sessions..."):
            # Extract subject ID from session ID (XXXX-XXXX is the subject ID)
            subject_id = '-'.join(session_id.split('-')[:2])
            if filter_by_mrn:
                mrn = mrn_crosswalk.get_mrn(subject_id) if mrn_crosswalk is not None else None
                if mrn is None:
                    continue
            else:
                mrn = None
            series_list: list[Series] = []
            series_dirs: list[tuple[str, Path]] = [(subdir.name, subdir) for subdir in (session_path / series_subdir_pattern).iterdir() if subdir.is_dir()]
            for series_id, series_path in series_dirs:
                dicoms: list[Path] = [dicom for dicom in (series_path / dicom_subdir_pattern).iterdir() if dicom.is_file()]
                if len(dicoms) == 0:
                    continue
                series_list.append(Series.from_dicom_series(dicoms, name=series_id) if load_dicom_fields else Series(name=series_id, n_scans=len(dicoms)))
            if len(series_list) == 0:
                continue
            session = Session(name=session_id, series=series_list, study_date=mrn_crosswalk.get_study_date(subject_id) if mrn_crosswalk is not None else None)
            sessions_list.append(session)
            # Check if subject already exists in subjects_list
            subject = next((s for s in subjects_list if s.name == subject_id), None)
            if subject is None:
                subject = Subject(name=subject_id, mrn=mrn, sessions=[session])
                subjects_list.append(subject)
            else:
                subject.sessions.append(session)
        subjects.subjects = subjects_list
        return subjects
    
    @classmethod
    def from_nested_dicom_dir(
        cls,
        dicom_root: Path | str,
        session_subdir_pattern: Path | str = "",
        series_subdir_pattern: Path | str = "",
        dicom_subdir_pattern: Path | str = "",
        mrn_crosswalk_path: Optional[Path | str] = None,
        filter_by_mrn: bool = True,
        sample: Optional[int] = None,
        load_dicom_fields: bool = False
    ) -> Self:
        """
        Assumes a directory structure of:
        Subject ID/
            Session ID/
                Series ID/
                    DICOM files...
        where Session ID is unique within each subject, and Subject ID is unique across all subjects.
        The pattern arguments append to the end of the respective subdirectory names, and can be used to filter for specific sessions/series/dicoms.
        
        Args:
            dicom_root: Root directory containing subject subdirectories.
            session_subdir_pattern: Optional pattern to append to the end of session subdirectory names for filtering (e.g. "SESSIONS/").
            series_subdir_pattern: Optional pattern to append to the end of series subdirectory names for filtering (e.g. "SCANS/").
            dicom_subdir_pattern: Optional pattern to append to the end of dicom subdirectory names for filtering (e.g. "DICOM/").
            mrn_crosswalk_path: Optional path to CSV file containing MRN crosswalk information.
            filter_by_mrn: Whether to filter subjects by MRN using the crosswalk. If True, only subjects with MRN information in the crosswalk will be included.
            sample: Optional integer to limit the number of subjects processed (for testing purposes).
            load_dicom_fields: Whether to load DICOM fields for each scan. If False, only the scan name will be loaded without the additional DICOM metadata.
        """
        dicom_root = Path(dicom_root)
        
        subjects: Self = cls()
        
        # Use the crosswalk to map subject ID with MRN and keep only those subjects going forward if filter_by_mrn is True
        mrn_crosswalk = None
        if mrn_crosswalk_path is not None:
            mrn_crosswalk = MRNCrosswalk.from_csv(mrn_crosswalk_path)
        if filter_by_mrn and mrn_crosswalk is None:
            raise ValueError("MRN crosswalk is required for filtering by MRN, but no mapping was found.")
        
        subjects_list: list[Subject] = []
        subject_dirs: list[tuple[str, Path]] = [(subdir.name, subdir) for subdir in dicom_root.iterdir() if subdir.is_dir()]
        if sample is not None:
            subject_dirs = subject_dirs[:sample]
        for subject_id, subject_path in track(subject_dirs, description="Processing subjects..."):
            if filter_by_mrn:
                mrn = mrn_crosswalk.get_mrn(subject_id) if mrn_crosswalk is not None else None
                if mrn is None:
                    continue
            else:
                mrn = None
            sessions_list: list[Session] = []
            session_dirs: list[tuple[str, Path]] = [(subdir.name, subdir) for subdir in (subject_path / session_subdir_pattern).iterdir() if subdir.is_dir()]
            for session_id, session_path in session_dirs:
                series_list: list[Series] = []
                series_dirs: list[tuple[str, Path]] = [(subdir.name, subdir) for subdir in (session_path / series_subdir_pattern).iterdir() if subdir.is_dir()]
                for series_id, series_path in series_dirs:
                    dicoms: list[Path] = [dicom for dicom in (series_path / dicom_subdir_pattern).iterdir() if dicom.is_file()]
                    if len(dicoms) == 0:
                        continue
                    series_list.append(Series(name=series_id, n_scans=len(dicoms)))
                if len(series_list) == 0:
                    continue
                session = Session(name=session_id, series=series_list, study_date=mrn_crosswalk.get_study_date(subject_id) if mrn_crosswalk is not None else None)
                sessions_list.append(session)
            if len(sessions_list) == 0:
                continue
            subject = Subject(name=subject_id, mrn=mrn, sessions=sessions_list)
            subjects_list.append(subject)
        subjects.subjects = subjects_list
        return subjects
        
    @classmethod
    def from_csv(cls, csv_path: Path | str) -> Self:
        df = pd.read_csv(csv_path)
        subjects_dict: dict[str, Subject] = {}
        for _, row in df.iterrows():
            subject_name = row['subject_name']
            mrn = row['mrn']
            session_name = row['session_name']
            study_date = pd.to_datetime(row['study_date'], errors='coerce').date() if not pd.isna(row['study_date']) else None
            series_name = row['series_name']
            series_description = row['series_description'] if not pd.isna(row['series_description']) else None
            series_date = pd.to_datetime(row['series_date'], errors='coerce').date() if not pd.isna(row['series_date']) else None
            n_scans = int(row['n_scans'])
            
            if subject_name not in subjects_dict:
                subjects_dict[subject_name] = Subject(name=subject_name, mrn=mrn, sessions=[])
            subject = subjects_dict[subject_name]
            
            session = next((s for s in subject.sessions if s.name == session_name), None)
            if session is None:
                session = Session(name=session_name, series=[], study_date=study_date)
                subject.sessions.append(session)
            
            series = Series(name=series_name, n_scans=n_scans, series_description=series_description, series_date=series_date)
            session.series.append(series)
        
        return cls(subjects=list(subjects_dict.values()))
    
    def to_csv(self, csv_path: Path | str) -> None:
        rows = []
        for subject in self.subjects:
            for session in subject.sessions:
                for series in session.series:
                    row = {
                        "subject_name": subject.name,
                        "mrn": subject.mrn,
                        "session_name": session.name,
                        "study_date": session.study_date,
                        "series_name": series.name,
                        "series_description": series.series_description,
                        "series_date": series.series_date,
                        "n_scans": series.n_scans
                    }
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def add_subject(self, subject: Subject) -> None:
        self.subjects.append(subject)
    
    def remove_subject(self, subject_name: str) -> None:
        self.subjects = [subject for subject in self.subjects if subject.name != subject_name]