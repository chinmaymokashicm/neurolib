"""
Work with BIDS data.
"""
from .dicom import convert_participant_mappings2table, ParticipantInfo

from typing import Optional
from pathlib import Path, PosixPath

from pydantic import BaseModel, DirectoryPath, FilePath
from rich.progress import track
import pandas as pd
from bids import BIDSLayout
from bids.layout import BIDSFile

class SelectBIDSFileInfo(BaseModel):
    """
    Selected BIDS information.
    """
    path: FilePath
    filename: str
    datatype: Optional[str]
    participant_id: Optional[str]
    session_id: Optional[str]
    run: Optional[str]
    series_number: Optional[str]
    acquisition_time: Optional[str]
    
    @classmethod
    def from_BIDSFile(cls, bids_file: BIDSFile) -> "SelectBIDSFileInfo":
        """
        Create a SelectBIDSFileInfo object from a BIDSFile object.
        """
        if not isinstance(bids_file, BIDSFile):
            raise ValueError(f"Expected a BIDSFile object, got {type(bids_file)}")
        path: str = bids_file.path
        filename: str = bids_file.filename
        entities: dict = {key: str(value) for key, value in bids_file.entities.items() if value is not None}
        datatype: str = entities.get("datatype", None)
        participant_id: str = entities.get("subject", None)
        session_id: str = entities.get("session", None)
        run: str = entities.get("run", None)
        series_number: str = entities.get("SeriesNumber", None)
        acquisition_time: str = entities.get("AcquisitionTime", None)
        # Convert acquisition time to a more readable format - HHMMSS
        acquisition_time = acquisition_time.replace(":", "").split(".")[0] if acquisition_time else None
        return cls(
            path=path,
            filename=filename,
            datatype=datatype,
            participant_id=participant_id,
            session_id=session_id,
            run=run,
            series_number=series_number,
            acquisition_time=acquisition_time,
        )
    
    @classmethod
    def from_BIDSLayout(cls, layout: BIDSLayout, filter: dict = {}, to_df: bool = False) -> list["SelectBIDSFileInfo"] | pd.DataFrame:
        """
        Create a list of SelectBIDSFileInfo objects from a BIDSLayout object.
        """
        bids_files: list[BIDSFile] = layout.get(**filter)
        table: list[SelectBIDSFileInfo] = [cls.from_BIDSFile(bids_file) for bids_file in bids_files]
        if to_df:
            return pd.DataFrame([info.model_dump() for info in table])
        return table
    
    @classmethod
    def merge_with_participants_info(cls, layout: BIDSLayout, participant_info: list[ParticipantInfo]) -> pd.DataFrame:
        """
        Merge the selected BIDS file information with the participants information.
        """
        df_participants: pd.DataFrame = pd.DataFrame(convert_participant_mappings2table(participant_info))
        df_bids_info: pd.DataFrame = cls.from_BIDSLayout(layout, to_df=True)
        df_mapped: pd.DataFrame = pd.merge(
            df_participants, 
            df_bids_info, 
            how="inner", 
            left_on=["participant_id", "bids_session_id", "series_number", "acquisition_time"], 
            right_on=["participant_id", "session_id", "series_number", "acquisition_time"]
        )
        df_mapped = df_mapped[["participant_id", "session_id_x", "bids_session_id", "scan_date", "scan_type", "run", "series_number", "acquisition_time", "filename", "scan_name"]].reset_index(drop=True)
        df_mapped.rename(columns={"session_id_x": "session_id", "filename": "bids_filename"}, inplace=True)
        df_mapped.head()
        return df_mapped