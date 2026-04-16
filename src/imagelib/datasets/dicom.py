"""
Subject -1:N-> Session -1:N-> Series -1:N-> Scan
"""
import re
from typing import Optional, Self, Any, Sequence
from pathlib import Path
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from pydantic import BaseModel, Field
import pandas as pd
from rich.progress import track
import pydicom as dicom

DICOM_TAGS: list[str | int | tuple[int, int]] = [
    "StudyDate",
    "StudyTime",
    "SeriesDate",
    "SeriesTime",
    "AcquisitionDate",
    "AcquisitionTime",
    "AcquisitionDateTime",
    "AccessionNumber",
    "InstitutionName",
    "SeriesInstanceUID",
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
    0x00211060,
    0x0019109D
    # (0x0021, 0x1060),
    # (0x0019, 0x109D),
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
    
    def get_subject_id(self, mrn: str) -> Optional[str]:
        for entry in self.entries:
            if entry.mrn == mrn:
                return entry.subject_id
        return None
    
    def get_study_date(self, subject_id: str) -> Optional[date]:
        for entry in self.entries:
            if entry.subject_id == subject_id:
                return entry.study_date
        return None

# def read_dicom_header(path: Path):
#     return dicom.dcmread(
#         path,
#         stop_before_pixels=True,
#         specific_tags=[
#             "SeriesInstanceUID",
#             "SeriesDescription",
#             "StudyDate",
#             "StudyTime",
#             "SeriesDate",
#             "SeriesTime",
#             "AcquisitionDate",
#             "AcquisitionTime",
#             "AcquisitionDateTime",
#             "AccessionNumber",
#             "InstitutionName",
#             "ProtocolName",
#             "SliceThickness",
#             "PixelSpacing",
#         ],
#         force=True,
#     )

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
    series_uid: Optional[str] = None
    series_description: Optional[str] = None
    study_date: Optional[date] = None
    series_date: Optional[date] = None
    series_time: Optional[str] = None
    protocol_name: Optional[str] = None
    slice_thickness: Optional[float] = None
    pixel_spacing: Optional[list[float]] = None
    institution_name: Optional[str] = None
    accession_number: Optional[str] = None
    
    @staticmethod
    def _parse_dicom_da(value: Any) -> Optional[date]:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return datetime.strptime(s[:8], "%Y%m%d").date()
        except ValueError:
            return None
    
    @staticmethod
    def extract_dicom_metadata(dicom_path: Path) -> Optional[dict]:
        try:
            dicom_data = dicom.dcmread(
                dicom_path,
                stop_before_pixels=True,
                # specific_tags=DICOM_TAGS,
                force=True,
            )
            metadata = {}
            for tag in DICOM_TAGS:
                if isinstance(tag, str):
                    metadata[tag] = getattr(dicom_data, tag, None)
                else:
                    data = dicom_data.get(tag, None)
                    if data is not None:
                        metadata[tag] = data.value
        except Exception:
            return None
        return metadata
        
    @staticmethod
    def resolve_study_date(metadata: dict) -> Optional[date]:
        """
        Study date can be stored in different DICOM tags depending on the scanner and acquisition protocol, 
        so we check multiple tags to try to resolve the study date.
        """
        date_tags = [0x00211060, 0x0019109D, 0x00291019, "AcquisitionDateTime", "AcquisitionDate", "StudyDate", "SeriesDate"]
        for tag in date_tags:
            if isinstance(tag, str):
                date_str = metadata.get(tag, None)
            else:
                data = metadata.get(tag, None)
                if data is None:
                    continue
                date_str = data if isinstance(data, str) else getattr(data, 'value', None)
            if date_str is not None:
                try:
                    if isinstance(date_str, str):
                        date_str = date_str.split(".")[0]  # Remove any fractional seconds if present
                        resolved_date = pd.to_datetime(date_str, errors='coerce').date()
                        if pd.notnull(resolved_date):
                            return resolved_date
                except Exception:
                    continue
        return None
    
    @classmethod
    def from_dicom_series(cls, dicom_series: Sequence[Path | str], name: Optional[str] = None) -> Self:
        # Extract series-level metadata from the first DICOM file in the series, and count the number of scans in the series
        if len(dicom_series) == 0:
            raise ValueError("DICOM series must contain at least one DICOM file.")
        for dicom_series_path in dicom_series:
            dicom_data = cls.extract_dicom_metadata(Path(dicom_series_path))
            if dicom_data is not None:
                break
        else:
            raise ValueError("Could not read metadata from any DICOM files in the series.")
        n_scans = len(dicom_series)
        series_description = dicom_data.get("SeriesDescription", None)
        study_date = cls.resolve_study_date(dicom_data)
        series_date = cls._parse_dicom_da(dicom_data.get("SeriesDate", None))
        series_time = dicom_data.get("AcquisitionTime", None)
        protocol_name = dicom_data.get("ProtocolName", None)
        slice_thickness_raw = dicom_data.get("SliceThickness", None)
        slice_thickness = float(slice_thickness_raw) if slice_thickness_raw is not None else None
        pixel_spacing_raw = dicom_data.get("PixelSpacing", None)
        pixel_spacing = [float(ps) for ps in pixel_spacing_raw] if pixel_spacing_raw is not None else None
        series_uid = dicom_data.get("SeriesInstanceUID", None)
        institution_name = dicom_data.get("InstitutionName", None)
        accession_number = dicom_data.get("AccessionNumber", None)
        
        return cls(
            name=name if name is not None else series_description if series_description is not None else series_uid if series_uid is not None else "Series",
            n_scans=n_scans,
            series_uid=series_uid,
            series_description=series_description,
            study_date=study_date,
            series_date=series_date,
            series_time=series_time,
            protocol_name=protocol_name,
            slice_thickness=slice_thickness,
            pixel_spacing=pixel_spacing,
            institution_name=institution_name,
            accession_number=accession_number,
        )
    
class Session(BaseModel):
    name: str
    series: list[Series] = Field(default_factory=list)
    study_date: Optional[date] = None
    study_time: Optional[str] = None
    accession_number: Optional[str] = None
    institution_name: Optional[str] = None
    
    @classmethod
    def from_series(cls, series_list: list[Series], name: Optional[str] = None) -> Self:
        if len(series_list) == 0:
            raise ValueError("Session must contain at least one series.")
        study_date = None
        study_time = None
        accession_number = None
        institution_name = None
        for series in series_list:
            if series.study_date is not None and study_date is None:
                study_date = series.study_date
            if series.series_time is not None and study_time is None:
                study_time = series.series_time
            if series.accession_number is not None and accession_number is None:
                accession_number = series.accession_number
            if series.institution_name is not None and institution_name is None:
                institution_name = series.institution_name
            if all(item is not None for item in [study_date, study_time, accession_number, institution_name]):
                break
        return cls(
            name=name if name is not None else f"Session_{study_date.isoformat()}" if study_date is not None else "Session",
            series=series_list,
            study_date=study_date,
            study_time=study_time,
            accession_number=accession_number,
            institution_name=institution_name,
        )
    
    def iterate(self):
        for series in self.series:
            yield series
    
class MRSubject(BaseModel):
    mrn: Optional[str] = None
    name: str
    sessions: list[Session] = Field(default_factory=list)
    
    def iterate(self):
        for session in self.sessions:
            yield session
            
    def remove_session(self, session_name: str) -> None:
        self.sessions = [session for session in self.sessions if session.name != session_name]
    
class MRSubjects(BaseModel):
    subjects: list[MRSubject] = Field(default_factory=list)
    
    def __str__(self) -> str:
        output = []
        for subject in self.subjects:
            output.append(f"Subject: {subject.name} (MRN: {subject.mrn})")
            for session in subject.sessions:
                output.append(f"  Session: {session.name} (Date: {session.study_date})")
                for series in session.series:
                    output.append(f"    Series: {series.name} (Description: {series.series_description}) - {series.n_scans} scans)")
        return "\n".join(output)
    
    def __getitem__(self, subject_name: str) -> Optional[MRSubject]:
        for subject in self.subjects:
            if subject.name == subject_name:
                return subject
        return None
    
    def iterate(self):
        for subject in self.subjects:
            yield subject
    
    def get_subject_by_mrn(self, mrn: str) -> Optional[MRSubject]:
        for subject in self.subjects:
            if subject.mrn == mrn:
                return subject
        return None
    
    @classmethod
    def from_flat_sessions_dicom_dir(
        cls,
        dicom_root: Path | str,
        series_subdir_pattern: Path | str = "",
        dicom_subdir_pattern: Path | str = "",
        mrn_crosswalk_path: Optional[Path | str] = None,
        filter_by_mrn: bool = True,
        sample: Optional[int] = None,
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
        
        # Use the crosswalk to map subject ID with MRN and keep only those subjects going forward if filter_by_mrn is True
        mrn_crosswalk = None
        if mrn_crosswalk_path is not None:
            mrn_crosswalk = MRNCrosswalk.from_csv(mrn_crosswalk_path)
        if filter_by_mrn and mrn_crosswalk is None:
            raise ValueError("MRN crosswalk is required for filtering by MRN, but no mapping was found.")
        
        # Identify each subdirectory in the root as a session if it is in the pattern of XXXX-XXXX-XXXX-XXXX (be liberal with the last set of digits - there could be additional characters there that we want to ignore, but we want to make sure we capture the first 16 digits which represent subject ID and session ID)
        subjects_dict: dict[str, list[Path]] = {}
        subjects: list[MRSubject] = []
        for session_dir in dicom_root.iterdir():
            if not session_dir.is_dir() or not re.match(r"^\d{4}-\d{4}-\d{4}-\d{4}.*", session_dir.name):
                continue
            subject_id = "-".join(session_dir.name.split("-")[:2])  # Capture the first 19 characters to include the full subject ID and session ID (16 characters + 3 dashes)
            if subject_id not in subjects_dict:
                subjects_dict[subject_id] = []
            subjects_dict[subject_id].append(session_dir)
        if sample is not None:
            subjects_dict = dict(list(subjects_dict.items())[:sample])
        for subject_id, session_dirs in track(subjects_dict.items(), description="Processing sessions...", total=len(subjects_dict)):
            sessions = []
            if filter_by_mrn:
                mrn = mrn_crosswalk.get_mrn(subject_id) if mrn_crosswalk is not None else None
                if mrn is None:
                    continue
            else:
                mrn = None
            for session_dir in session_dirs:
                series_path = session_dir / series_subdir_pattern
                if not series_path.exists():
                    continue
                series_dirs = [d for d in series_path.iterdir() if d.is_dir()]
                series_list = []
                for series_dir in series_dirs:
                    dicom_files = list((series_dir / dicom_subdir_pattern).rglob("*.dcm"))
                    if len(dicom_files) == 0:
                        continue
                    try:
                        series_obj = Series.from_dicom_series(dicom_series=dicom_files, name=series_dir.name)
                        series_list.append(series_obj)
                    except Exception as e:
                        # print(f"Failed to create Series object from DICOM files in {series_dir}, skipping... Error: {e}")
                        raise e
                        continue
                if len(series_list) == 0:
                    continue
                session_obj = Session.from_series(series_list=series_list, name=session_dir.name)
                sessions.append(session_obj)
            if len(sessions) == 0:
                continue
            subject_obj = MRSubject(name=subject_id, mrn=mrn, sessions=sessions)
            subjects.append(subject_obj)
        return cls(subjects=subjects)
    
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
        """
        dicom_root = Path(dicom_root)
        
        # Use the crosswalk to map subject ID with MRN and keep only those subjects going forward if filter_by_mrn is True
        mrn_crosswalk = None
        if mrn_crosswalk_path is not None:
            mrn_crosswalk = MRNCrosswalk.from_csv(mrn_crosswalk_path)
        if filter_by_mrn and mrn_crosswalk is None:
            raise ValueError("MRN crosswalk is required for filtering by MRN, but no mapping was found.")
        
        subjects_dict: dict[str, list[Path]] = {}
        subjects: list[MRSubject] = []
        for subject_dir in dicom_root.iterdir():
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            if subject_id not in subjects_dict:
                subjects_dict[subject_id] = []
            subjects_dict[subject_id].append(subject_dir)
        if sample is not None:
            subjects_dict = dict(list(subjects_dict.items())[:sample])
        for subject_id, subject_dirs in subjects_dict.items():
            sessions = []
            if filter_by_mrn:
                mrn = mrn_crosswalk.get_mrn(subject_id) if mrn_crosswalk is not None else None
                if mrn is None:
                    continue
            else:
                mrn = None
            for subject_dir in subject_dirs:
                # session_dirs = [d for d in (subject_dir / session_subdir_pattern).iterdir() if d.is_dir()]
                session_path = subject_dir / session_subdir_pattern
                if not session_path.exists():
                    continue
                session_dirs = [d for d in session_path.iterdir() if d.is_dir()]
                for session_dir in session_dirs:
                    series_path = session_dir / series_subdir_pattern
                    if not series_path.exists():
                        continue
                    series_dirs = [d for d in series_path.iterdir() if d.is_dir()]
                    series_list = []
                    for series_dir in series_dirs:
                        dicom_files = list((series_dir / dicom_subdir_pattern).rglob("*.dcm"))
                        if len(dicom_files) == 0:
                            continue
                        try:
                            series_obj = Series.from_dicom_series(dicom_series=dicom_files, name=series_dir.name)
                            series_list.append(series_obj)
                        except Exception:
                            continue
                    if len(series_list) == 0:
                        continue
                    session_obj = Session.from_series(series_list=series_list, name=session_dir.name)
                    sessions.append(session_obj)
            if len(sessions) == 0:
                continue
            subject_obj = MRSubject(name=subject_id, mrn=mrn, sessions=sessions)
            subjects.append(subject_obj)
        return cls(subjects=subjects)
        
    @classmethod
    def from_csv(cls, csv_path: Path | str) -> Self:
        df = pd.read_csv(csv_path)
        subjects_dict: dict[str, dict[str, list[Series]]] = {}
        subject_to_mrn: dict[str, str] = {}
        for _, row in df.iterrows():
            subject_name = row['subject_name']
            mrn = row['mrn']
            session_name = row['session_name']
            study_date = pd.to_datetime(row['study_date'], errors='coerce').date() if pd.notnull(row['study_date']) else None
            series_name = row['series_name']
            series_description = row['series_description']
            series_date = pd.to_datetime(row['series_date'], errors='coerce').date() if pd.notnull(row['series_date']) else None
            n_scans = int(row['n_scans']) if pd.notnull(row['n_scans']) else 0
            
            series_obj = Series(
                name=series_name,
                n_scans=n_scans,
                series_description=series_description,
                series_date=series_date,
                study_date=study_date,
            )
            if subject_name not in subjects_dict:
                subjects_dict[subject_name] = {}
            if session_name not in subjects_dict[subject_name]:
                subjects_dict[subject_name][session_name] = []
            subjects_dict[subject_name][session_name].append(series_obj)
            subject_to_mrn[subject_name] = str(mrn)
        subjects = []
        for subject_name, sessions_dict in subjects_dict.items():
            sessions = []
            for session_name, series_list in sessions_dict.items():
                session_obj = Session.from_series(series_list=series_list, name=session_name)
                sessions.append(session_obj)
            subject_obj = MRSubject(name=subject_name, mrn=subject_to_mrn.get(subject_name, None), sessions=sessions)
            subjects.append(subject_obj)
        return cls(subjects=subjects)
    
    def to_csv(self, csv_path: Path | str) -> None:
        rows = []
        for subject in self.subjects:
            for session in subject.sessions:
                for series in session.series:
                    row = {
                        'subject_name': subject.name,
                        'mrn': subject.mrn,
                        'session_name': session.name,
                        'study_date': session.study_date.isoformat() if session.study_date else None,
                        'series_name': series.name,
                        'series_description': series.series_description,
                        'series_date': series.series_date.isoformat() if series.series_date else None,
                        'n_scans': series.n_scans,
                    }
                    rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

    def add_subject(self, subject: MRSubject) -> None:
        self.subjects.append(subject)
    
    def remove_subject(self, subject_name: str) -> None:
        self.subjects = [subject for subject in self.subjects if subject.name != subject_name]