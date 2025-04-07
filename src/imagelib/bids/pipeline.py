"""
Code related to BIDS pipelines.
"""
# from __future__ import annotations

from ..datasets.bids import DatasetDescription, GeneratedBy, BIDSTree
from ..helpers.data import clean_dict_values
from .processes.base import BIDSProcess, BIDSProcessResults

from pathlib import Path, PosixPath
from typing import Optional, Any
from collections.abc import Callable
import json
from copy import deepcopy

from bids import BIDSLayout
from bids.layout import BIDSFile, parse_file_entities
from bids.layout.writing import build_path
from pydantic import BaseModel, DirectoryPath, FilePath, Field, field_validator
from rich.progress import track

def get_new_pipeline_derived_filename(bids_file: BIDSFile | str, bids_layout: BIDSLayout, pipeline_name: str, desc: Optional[str] = None, extension: Optional[str] = None, suffix: Optional[str] = None, **other_entities) -> str:
    """
    Generate BIDS filename from a BIDS file for a pipeline.
    """
    if isinstance(bids_file, BIDSFile):
        bids_file: str = bids_file.path
    
    entities: dict = parse_file_entities(bids_file)
    entities = clean_dict_values(entities, [">", "<", "'", '"'])
    
    if desc:
        entities["desc"] = desc
    if suffix:
        entities["suffix"] = suffix
    if extension:
        entities["extension"] = extension
    
    if other_entities:
        entities.update(other_entities)
        
    if entities["extension"] is None:
        raise ValueError(f"Extension cannot be None for {bids_file}")
    
    if extension[0] != ".":
        extension = "." + extension
    
    path_patterns = [
        "sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_rec-{rec}][_run-{run}][_desc-{desc}][_{suffix}]{extension}",
        "sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_run-{run}][_desc-{desc}][_{suffix}]{extension}",
        "sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_desc-{desc}][_{suffix}]{extension}",
        # "sub-{subject}/ses-{session}/{datatype}/sub-{subject}_ses-{session}_{suffix}{extension}",
        # "sub-{subject}/{datatype}/sub-{subject}_desc-{desc}_{suffix}{extension}",
        # "sub-{subject}/{datatype}/sub-{subject}_{suffix}{extension}"
    ]
    
    derivative_filename: str = build_path(entities, path_patterns, strict=False)
    return str(Path(bids_layout.root) / "derivatives" / pipeline_name / derivative_filename)

class BIDSPipeline(BaseModel):
    """
    Pipeline dataset in BIDS-format generated from original BIDS dataset.
    """
    name: str = Field(title="Pipeline name", description="Name of the pipeline")
    description: Optional[str] = Field(title="Description", description="Description of the pipeline", default=None)
    bids_root: DirectoryPath = Field(title="BIDS root directory", description="Root directory of the BIDS dataset")
    tree: BIDSTree = Field(title="Pipeline tree", description="Pipeline tree")
    # bids_filters: dict = Field(title="BIDS filters", description="BIDS filters", default={})
    processes: list[BIDSProcess] = Field(title="Processes", description="Processes in the pipeline", default=[])
    # is_chain: bool = Field(title="Is chain", description="Is the pipeline a chain of processes", default=False)
    overwrite_files: bool = Field(title="Overwrite files", description="Overwrite files if they already exist", default=False)
            
    @classmethod
    def from_path(cls, name: str, bids_root: str | DirectoryPath) -> "BIDSPipeline":
        """
        Create a BIDSPipeline object from a BIDS root directory.
        """
        bids_root: PosixPath = Path(bids_root)
        tree: BIDSTree = BIDSTree.from_path(bids_root / "derivatives" / name)
        return cls(name=name, bids_root=bids_root, tree=tree)
    
    def create_tree(self):
        """
        Create the pipeline tree.
        """
        derivatives_dir: PosixPath = self.bids_root / "derivatives" / self.name
        if derivatives_dir.exists():
            print(f"Directory {derivatives_dir} already exists. Skipping creation.")
            return
        
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        self.tree.set_default_values(self.name)
        self.tree.create(derivatives_dir)
    
    def add_process(self, process: BIDSProcess):
        """
        Add a process to the pipeline.
        """
        self.processes.append(process)
        
    def execute(self):
        """
        Execute the pipeline.
        """
        for process in self.processes:
            bids_filters: dict = deepcopy(process.bids_filters)
            if "return_type" not in bids_filters or not bids_filters["return_type"]:
                bids_filters["return_type"] = "file"
            layout: BIDSLayout = BIDSLayout(self.bids_root, derivatives=True)
            filepaths: list[str] = layout.get(**bids_filters)
            # print(f"Executing {process.name} using BIDS filters: {bids_filters} on {len(filepaths)} files.")
            process.kwargs["layout"] = layout
            process.kwargs["pipeline_name"] = self.name
            process.kwargs["overwrite"] = self.overwrite_files
            for filepath in track(filepaths, description=f"Processing {process.name} using BIDS filters: {bids_filters} on {len(filepaths)} files"):
                process.kwargs["input_filepath"] = filepath
                process.execute()
    
    def to_dict(self) -> dict:
        """
        Convert the pipeline to a dictionary.
        """
        return {
            "name": self.name,
            "bids_root": str(self.bids_root),
            "tree": self.tree.model_dump(mode="json"),
            # "bids_filters": self.bids_filters,
            "processes": [process.to_dict() for process in self.processes],
        }
        
    def export_to_json(self) -> None:
        """
        Export the pipeline to a JSON file.
        """
        save_dir: PosixPath = self.bids_root / "derivatives" / self.name
        with open(save_dir / f"{self.name}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)