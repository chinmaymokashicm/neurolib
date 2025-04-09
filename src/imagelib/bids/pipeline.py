"""
Code related to BIDS pipelines.
"""
# from __future__ import annotations

from ..datasets.bids import DatasetDescription, GeneratedBy, BIDSTree
from ..helpers.data import clean_dict_values
from ..helpers.generate import generate_id
from .processes.base import BIDSProcess, BIDSProcessResults, BIDSProcessExec

from pathlib import Path, PosixPath
from typing import Optional, Any
from collections.abc import Callable
import json, os
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
    id: str = Field(title="ID. The unique identifier of the pipeline.", default_factory=lambda: generate_id("PL", 10, "-"))
    name: str = Field(title="Pipeline name", description="Name of the pipeline")
    description: Optional[str] = Field(title="Description", description="Description of the pipeline", default=None)
    process_execs: list[BIDSProcessExec] = Field(title="Process executions.", description="Process executions in the pipeline", default_factory=list)
    overwrite: bool = Field(title="Overwrite files", description="Overwrite files if they already exist", default=False)
    
    @classmethod
    def from_path(cls, name: str, bids_root: str | DirectoryPath) -> "BIDSPipeline":
        """
        Create a BIDSPipeline object from a BIDS root directory.
        """
        bids_root: PosixPath = Path(bids_root)
        tree: BIDSTree = BIDSTree.from_path(bids_root / "derivatives" / name)
        return cls(name=name, bids_root=bids_root, tree=tree)
    
    def set_values_from_env(self):
        """
        Set values from environment variables.
        """
        self.id = os.getenv("PIPELINE_ID", self.id)
        self.pipeline_name = os.getenv("PIPELINE_NAME", self.name)
        self.overwrite = os.getenv("OVERWRITE", self.overwrite)
    
    def create_trees(self):
        """
        Create the pipeline trees.
        """
        all_bids_roots: list[PosixPath] = []
        for process_exec in self.process_execs:
            bids_roots: list[PosixPath] = process_exec.bids_roots
            for bids_root in bids_roots:
                if bids_root not in all_bids_roots:
                    all_bids_roots.append(bids_root)
                    
        for bids_root in all_bids_roots:
            bids_root: PosixPath = Path(bids_root)
            tree: BIDSTree = BIDSTree()
            tree.set_default_values(self.name)
            tree.create(bids_root / "derivatives" / self.name)
    
    def add_process_exec(self, process_exec: BIDSProcessExec):
        """
        Add a process execution to the pipeline.
        """
        self.process_execs.append(process_exec)
        
    def execute(self):
        """
        Execute the pipeline.
        """
        for process_exec in self.process_execs:
            print(f"Executing {process_exec.process.name}...")
            process_exec.pipeline_id = self.id
            process_exec.pipeline_name = self.name
            process_exec.execute()
            
    def to_dict(self) -> dict:
        """
        Convert the pipeline to a dictionary.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "process_execs": [process_exec.to_dict() for process_exec in self.process_execs],
            "overwrite": self.overwrite
        }
        
    def export_to_json(self) -> None:
        """
        Export the pipeline to a JSON file.
        """
        save_dir: PosixPath = self.bids_root / "derivatives" / self.name
        with open(save_dir / f"{self.name}.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)