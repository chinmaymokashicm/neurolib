"""
Code related to BIDS pipelines.
"""
# from __future__ import annotations

from .base import DatasetDescription, GeneratedBy
from ..helpers.data import clean_dict_values

from pathlib import Path, PosixPath
from typing import Optional

from bids import BIDSLayout
from bids.layout import BIDSFile, parse_file_entities
from bids.layout.writing import build_path
from pydantic import BaseModel, DirectoryPath, FilePath, Field

def get_new_pipeline_derived_filename(bids_file: BIDSFile | str, bids_layout: BIDSLayout, pipeline_name: str, desc: Optional[str] = None, extension: Optional[str] = None) -> str:
    """
    Generate BIDS filename from a BIDS file for a pipeline.
    """
    if isinstance(bids_file, BIDSFile):
        bids_file: str = bids_file.path
    # bids_filename: str = Path(bids_file).name
    
    entities: dict = parse_file_entities(bids_file)
    entities = clean_dict_values(entities, [">", "<", "'", '"'])
    
    if desc:
        entities["desc"] = desc
    if extension:
        entities["extension"] = extension
        
    if entities["extension"] is None:
        raise ValueError(f"Extension cannot be None for {bids_file}")
    
    if extension[0] != ".":
        extension = "." + extension
    
    path_patterns = [
        "sub-{subject}[/ses-{session}]/{datatype}/sub-{subject}[_ses-{session}][_desc-{desc}][_{suffix}]{extension}",
        # "sub-{subject}/ses-{session}/{datatype}/sub-{subject}_ses-{session}_{suffix}{extension}",
        # "sub-{subject}/{datatype}/sub-{subject}_desc-{desc}_{suffix}{extension}",
        # "sub-{subject}/{datatype}/sub-{subject}_{suffix}{extension}"
    ]
    
    derivative_filename: str = build_path(entities, path_patterns, strict=False)
    print(f"Derivative filename: {derivative_filename}")
    return str(Path(bids_layout.root) / "derivatives" / pipeline_name / derivative_filename)

class BIDSPipelineTree(BaseModel):
    """
    BIDS pipeline tree. Define the structure of the pipeline dataset.
    """
    dataset_description: Optional[DatasetDescription] = Field(title="Dataset description", description="Dataset description", default=None)
    readme_text: Optional[str] = Field(title="Readme text", description="Readme text", default=None)
    citation_text: Optional[str] = Field(title="Citation text", description="Citation text", default=None)
    changes_text: Optional[str] = Field(title="Changes text", description="Changes text", default=None)
    license_text: Optional[str] = Field(title="License text", description="License text", default=None)
    
    def set_default_values(self, name: str):
        if not self.dataset_description:
            self.dataset_description = DatasetDescription(Name=name, GeneratedBy=[GeneratedBy(Name="BIDS Pipeline")])
        if not self.readme_text:
            self.readme_text = f"# {name} Pipeline\n\n"
        if not self.citation_text:
            self.citation_text = f"# {name}\n\n"
        if not self.changes_text:
            self.changes_text = f"# {name}\n\n"
        if not self.license_text:
            self.license_text = f"# {name}\n\n"
            
    @classmethod
    def from_path(cls, dirpath: DirectoryPath) -> "BIDSPipelineTree":
        """
        Load pipeline tree from a directory.
        """
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

class BIDSPipeline(BaseModel):
    """
    Pipeline dataset in BIDS-format generated from original BIDS dataset.
    """
    name: str = Field(title="Pipeline name", description="Name of the pipeline")
    bids_root: DirectoryPath = Field(title="BIDS root directory", description="Root directory of the BIDS dataset")
    tree: BIDSPipelineTree = Field(title="Pipeline tree", description="Pipeline tree")
    
    def create_tree(self):
        """
        Create the pipeline tree.
        """
        derivatives_dir: PosixPath = self.bids_root / "derivatives" / self.name
        derivatives_dir.mkdir(parents=True, exist_ok=True)
        self.tree.set_default_values(self.name)
        with open(derivatives_dir / "dataset_description.json", "w") as f:
            f.write(self.tree.dataset_description.model_dump_json(indent=4))
        with open(derivatives_dir / "README", "w") as f:
            f.write(self.tree.readme_text)
        with open(derivatives_dir / "CITATION", "w") as f:
            f.write(self.tree.citation_text)
        with open(derivatives_dir / "CHANGES", "w") as f:
            f.write(self.tree.changes_text)
        with open(derivatives_dir / "LICENSE", "w") as f:
            f.write(self.tree.license_text)
            
    @classmethod
    def from_path(cls, name: str, bids_root: DirectoryPath) -> "BIDSPipeline":
        """
        Create a BIDSPipeline object from a BIDS root directory.
        """
        tree: BIDSPipelineTree = BIDSPipelineTree.from_path(bids_root / "derivatives" / name)
        return cls(name=name, bids_root=bids_root, tree=tree)
            
class BIDSPreprocessingPipeline(BIDSPipeline):
    def t1w_preprocessing(self, bids_file: BIDSFile | str, extension: Optional[str] = None) -> str:
        """
        Preprocess T1w image.
        """
        pass