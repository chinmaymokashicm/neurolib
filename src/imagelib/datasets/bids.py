"""
Work with BIDS data.
"""
from .dicom import Participants

from pathlib import Path, PosixPath
from typing import Optional, Any
import json

from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator, ConfigDict
from rich.progress import track
import pandas as pd
from bids import BIDSLayout 
from bids.layout import BIDSFile

class GeneratedBy(BaseModel):
    Name: str = Field(title="Name", description="Name of the software used to generate the dataset")
    Version: Optional[str] = Field(title="Version", description="Version of the software used to generate the dataset", default=None)
    Description: Optional[str] = Field(title="Description", description="Description of the software used to generate the dataset", default=None)
    CodeURL: Optional[str] = Field(title="Code URL", description="URL to the code used to generate the dataset", default=None)
    Container: Optional[dict[str, str]] = Field(title="Container", description="Container information", default={})

class DatasetDescription(BaseModel):
    """
    Dataset description.
    """
    Name: str = Field(title="Name", description="Name of the dataset")
    BIDSVersion: str = Field(title="BIDS version", description="BIDS version", default="0.0.0")
    HEDVersion: Optional[str | list[str]] = Field(title="HED version", description="HED version", default="")
    DatasetLinks: Optional[str] = Field(title="Dataset links", description="Links to the dataset", default="")
    DatasetType: Optional[str] = Field(title="Dataset type", description="Type of the dataset", default="raw")
    License: Optional[str] = Field(title="License", description="License information", default="CC0")
    Authors: Optional[list[str]] = Field(title="Authors", description="Authors", default=[])
    Acknowledgements: Optional[str] = Field(title="Acknowledgements", description="Acknowledgements", default="")
    HowToAcknowledge: Optional[str] = Field(title="How to acknowledge", description="How to acknowledge the dataset", default="")
    Funding: Optional[list[str]] = Field(title="Funding", description="Funding information", default=[])
    EthicsApprovals: Optional[list[str]] = Field(title="Ethics approvals", description="Ethics approvals", default=[])
    ReferencesAndLinks: Optional[list[str]] = Field(title="References and links", description="References and links", default=[])
    DatasetDOI: Optional[str] = Field(title="Dataset DOI", description="Dataset DOI", default="")
    GeneratedBy: Optional[list[Any]] = Field(title="Generated by", description="Generated by.", default=[
        GeneratedBy(Name="BIDS Pipeline", Version="0.0.1", Description="BIDS Pipeline", CodeURL="")
    ])
    SourceDatasets: Optional[list[dict[str, str]]] = Field(title="Source datasets", description="Source datasets", default=[])
    
    @field_validator("GeneratedBy", mode="before")
    def set_GeneratedBy(cls, v):
        if isinstance(v, dict):
            return [GeneratedBy(**v)]
        elif isinstance(v, list):
            return [GeneratedBy(**item) for item in v]
        return v
    
    @classmethod
    def from_file(cls, path: str) -> "DatasetDescription":
        """
        Load dataset description from a JSON file.
        """
        with open(path, "r") as f:
            data: dict = json.load(f)
        return cls(**data)
    
def update_participants_json(participants_dict: Optional[dict], overwrite: bool = False, **kwargs) -> dict[str, Any]:
    """
    Add/update participants.json information.
    """
    default_participants_dict: dict[str, Any] = {
        "age": {
            "Description": "age of the participant",
            "Units": "year"
        },
        "sex": {
            "Description": "sex of the participant as reported by the participant",
            "Levels": {
                "M": "male",
                "F": "female"
            }
        },
        "handedness": {
            "Description": "handedness of the participant as reported by the participant",
            "Levels": {
                "left": "left",
                "right": "right"
            }
        },
        "group": {
            "Description": "experimental group the participant belonged to",
            "Levels": {
                "read": "participants who read an inspirational text before the experiment",
                "write": "participants who wrote an inspirational text before the experiment"
            }
        }
    }
    if participants_dict is None:
        participants_dict = default_participants_dict
    
    for key, value in kwargs.items():
        if overwrite or key not in participants_dict:
            participants_dict[key] = value
    return participants_dict

class BIDSTree(BaseModel):
    """
    BIDS tree. Define the structure of the pipeline dataset.
    """
    dataset_description: Optional[DatasetDescription] = Field(title="Dataset description", description="Dataset description", default=None)
    readme_text: Optional[str] = Field(title="Readme text", description="Readme text", default=None)
    citation_text: Optional[str] = Field(title="Citation text", description="Citation text", default=None)
    changes_text: Optional[str] = Field(title="Changes text", description="Changes text", default=None)
    license_text: Optional[str] = Field(title="License text", description="License text", default=None)
            
    @classmethod
    def from_path(cls, dirpath: str | DirectoryPath) -> "BIDSTree":
        """
        Load pipeline tree from a directory.
        """
        dirpath: PosixPath = Path(dirpath)
        dataset_description: DatasetDescription = DatasetDescription.from_file(dirpath / "dataset_description.json")
        with open(dirpath / "README", "r") as f:
            readme_text: str = f.read()
        with open(dirpath / "CITATION", "r") as f:
            citation_text: str = f.read()
        with open(dirpath / "CHANGES", "r") as f:
            changes_text: str = f.read()
        with open(dirpath / "LICENSE", "r") as f:
            license_text: str = f.read()
        return cls(
            dataset_description=dataset_description,
            readme_text=readme_text,
            citation_text=citation_text,
            changes_text=changes_text,
            license_text=license_text
        )
    
    def to_dict(self) -> dict:
        """
        Convert the pipeline tree to a dictionary.
        """
        return {
            "dataset_description": self.dataset_description.model_dump(mode="json"),
            "readme_text": self.readme_text,
            "citation_text": self.citation_text,
            "changes_text": self.changes_text,
            "license_text": self.license_text
        }
    
    def set_default_values(self, name: str):
        if not self.dataset_description:
            self.dataset_description = DatasetDescription(Name=name)
        if not self.readme_text:
            self.readme_text = f"# {name} Pipeline\n\n"
        if not self.citation_text:
            self.citation_text = f"# {name}\n\n"
        if not self.changes_text:
            self.changes_text = f"# {name}\n\n"
        if not self.license_text:
            self.license_text = f"# {name}\n\n"
            
    def create(self, dirpath: str | DirectoryPath, overwrite: bool = False) -> None:
        """
        Create the pipeline tree.
        """
        dirpath: PosixPath = Path(dirpath)
        if (dirpath / "dataset_description.json").exists() and not overwrite:
            print(f"Directory {dirpath} already exists. Skipping creation.")
            return
        if self.dataset_description is None:
            self.set_default_values(name=dirpath.name)
        dirpath.mkdir(parents=True, exist_ok=True)
        with open(dirpath / "dataset_description.json", "w") as f:
            f.write(self.dataset_description.model_dump_json(indent=4))
        with open(dirpath / "README", "w") as f:
            f.write(self.readme_text)
        with open(dirpath / "CITATION", "w") as f:
            f.write(self.citation_text)
        with open(dirpath / "CHANGES", "w") as f:
            f.write(self.changes_text)
        with open(dirpath / "LICENSE", "w") as f:
            f.write(self.license_text)

class ParticipantDemographyMetric(BaseModel):
    """
    Demographic metric about a participant.
    """
    name: str = Field(title="Name", description="Name of the metric")
    description: Optional[str] = Field(title="Description", description="Description of the metric", default=None)
    units: Optional[str] = Field(title="Units", description="Units of the metric", default=None)
    value: str = Field(title="Value", description="Value of the metric")

class BIDSParticipant(BaseModel):
    """
    Information about a participant in the BIDS dataset.
    """
    subject_id: str = Field(title="Subject ID", description="Subject ID")
    demographic_metrics: list[ParticipantDemographyMetric] = Field(title="Demographic metrics", description="Demographic metrics", default_factory=list)
    
class BIDSParticipants(BaseModel):
    """
    Information about participants in the BIDS dataset.
    """
    participants: list[BIDSParticipant] = Field(title="Participants", description="List of participants", default_factory=list)
    
    @classmethod
    def from_path(cls, path: str | PosixPath, metric_colnames: Optional[list[str]] = []) -> "BIDSParticipants":
        """
        Load participants information from a file.
        """
        df_participants: pd.DataFrame = pd.read_csv(path, sep="\t")
        participants: list[BIDSParticipant] = []
        if metric_colnames is None:
            metric_colnames = [colname for colname in df_participants.columns.tolist() if colname not in ["participant_id", "scan_date", "scan_type"]]
        for _, row in df_participants.iterrows():
            demographic_metrics: list[ParticipantDemographyMetric] = []
            for col in df_participants.columns:
                if col not in metric_colnames:
                    demographic_metrics.append(ParticipantDemographyMetric(name=col, value=row[col]))
            participant: BIDSParticipant = BIDSParticipant(subject_id=row["participant_id"], demographic_metrics=demographic_metrics)
            participants.append(participant)

class SelectBIDSDatasetInfo(BaseModel):
    """
    Selected BIDS dataset information.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    bids_root: DirectoryPath
    bids_files: list[BIDSFile]
    dataset_description: DatasetDescription
    derivatives: Optional[dict[str, "SelectBIDSDatasetInfo"]] = Field(title="Derivatives", description="Derivatives information", default_factory=dict)
    participants: Optional[BIDSParticipants] = Field(title="Participants", description="Participants information", default=None)

    @classmethod
    def from_path(cls, path: str | DirectoryPath, get_derivatives: bool = True) -> "SelectBIDSDatasetInfo":
        """
        Load BIDS dataset information from a directory.
        """
        bids_root: DirectoryPath = Path(path)
        layout: BIDSLayout = BIDSLayout(bids_root, derivatives=get_derivatives)
        dataset_description: DatasetDescription = DatasetDescription.from_file(bids_root / "dataset_description.json")
        bids_files: list[BIDSFile] = layout.get()
        derivatives: dict[str, str] = {}
        for sub_path in layout.derivatives:
            derivative_name: str = Path(sub_path).name
            derivative_path: DirectoryPath = bids_root / sub_path
            derivatives[derivative_name] = SelectBIDSDatasetInfo.from_path(derivative_path, get_derivatives=get_derivatives)
        
        if Path(bids_root / "participants.tsv").exists():
            participants: BIDSParticipants = BIDSParticipants.from_path(bids_root / "participants.tsv")
        
        return cls(
            bids_root=bids_root,
            bids_files=bids_files,
            dataset_description=dataset_description,
            derivatives=derivatives,
            participants=participants
        )
        
    def to_dict(self) -> dict:
        """
        Convert the dataset information to a dictionary.
        """
        return {
            "bids_root": str(self.bids_root),
            "bids_files": [str(Path(bids_file.path).relative_to(self.bids_root)) for bids_file in self.bids_files],
            "dataset_description": self.dataset_description.model_dump(mode="json"),
            "derivatives": {key: value.to_dict() for key, value in self.derivatives.items()},
            "participants": self.participants.model_dump(mode="json") if self.participants else None
        }
        
    def to_json(self, path: str | DirectoryPath) -> None:
        """
        Save the dataset information to a JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


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
    series_description: Optional[str]
    series_number: Optional[str]
    acquisition_time: Optional[str]
    
    @classmethod
    def from_BIDSFile(cls, bids_file: str | BIDSFile) -> "SelectBIDSFileInfo":
        """
        Create a SelectBIDSFileInfo object from a BIDSFile object.
        """
        # if not isinstance(bids_file, BIDSFile):
        #     raise ValueError(f"Expected a BIDSFile object, got {type(bids_file)}")
        bids_file: BIDSFile = bids_file if isinstance(bids_file, BIDSFile) else BIDSFile(bids_file)
        path: str = bids_file.path
        filename: str = bids_file.filename
        entities: dict = {key: str(value) for key, value in bids_file.entities.items() if value is not None}
        datatype: str = entities.get("datatype", None)
        participant_id: str = entities.get("subject", None)
        session_id: str = entities.get("session", None)
        run: str = entities.get("run", "0")
        series_description: str = entities.get("SeriesDescription", None)
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
            series_description=series_description,
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
    def merge_with_participants_info(cls, df_or_layout: pd.DataFrame | BIDSLayout, participants: Participants) -> pd.DataFrame:
        """
        Merge the selected BIDS file information with the participants information.
        """
        df_participants: pd.DataFrame = participants.to_table(to_df=True)
        if isinstance(df_or_layout, pd.DataFrame):
            df_bids_info: pd.DataFrame = df_or_layout
        else:
            layout: BIDSLayout = df_or_layout
            df_bids_info: pd.DataFrame = cls.from_BIDSLayout(layout, to_df=True)
        df_mapped: pd.DataFrame = pd.merge(
            df_participants, 
            df_bids_info, 
            how="outer", 
            left_on=["participant_id", "bids_session_id", "series_description", "series_number", "acquisition_time"], 
            right_on=["participant_id", "session_id", "series_description", "series_number", "acquisition_time"]
        )
        df_mapped = df_mapped[["participant_id", "session_id_x", "bids_session_id", "scan_date", "scan_type", "run", "series_description", "series_number", "acquisition_time", "filename", "scan_name"]].reset_index(drop=True)
        df_mapped.rename(columns={"session_id_x": "session_id", "filename": "bids_filename"}, inplace=True)
        df_mapped.head()
        return df_mapped