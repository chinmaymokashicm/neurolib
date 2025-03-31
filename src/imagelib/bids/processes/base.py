from typing import Optional, Any, Callable
from pathlib import Path, PosixPath
import functools, traceback, json

from pydantic import BaseModel, Field, field_validator
from bids import BIDSLayout

class BIDSProcess(BaseModel):
    """
    Define a process in the BIDS pipeline.
    """
    name: str = Field(title="Name", description="Name of the process")
    description: Optional[str] = Field(title="Description", description="Description of the process", default=None)
    logic: Callable = Field(title="Logic", description="Logic of the process. A callable object.")
    bids_filters: dict = Field(title="BIDS filters", description="BIDS filters to apply to the input files", default={})
    kwargs: dict = Field(title="Keyword arguments", description="Keyword arguments for the process", default={})
    
    def execute(self) -> Optional[str]:
        """
        Execute the process.
        """
        return self.logic(**self.kwargs)
    
    def to_dict(self) -> dict:
        """
        Convert the process to a dictionary.
        """
        kwargs: dict = {}
        for key, value in self.kwargs.items():
            if isinstance(value, PosixPath):
                value = str(value)
            elif isinstance(value, BIDSLayout):
                value = value.root
            kwargs[key] = value
        return {
            "name": self.name,
            "description": self.description,
            "kwargs": kwargs
        }
    
    def set_input_filepaths(self, layout: BIDSLayout, bids_filters: dict):
        """
        Set the input filepaths for the process.
        """
        self.input_filepaths = layout.get(return_type="file", **bids_filters)

class BIDSProcessSummarySidecar(BaseModel):
    """
    Sidecars that summarize the process.
    """
    process_id: Optional[str] = Field(title="Process ID", description="Unique identifier for the process", default=None)
    pipeline_id: Optional[str] = Field(title="Pipeline ID", description="Unique identifier for the pipeline", default=None)
    name: str = Field(title="Name", description="Name of the sidecar file to be saved (without extension)")
    description: Optional[str] = Field(title="Description", description="Description of the process", default=None)
    pipeline_name: str = Field(title="Pipeline name", description="Name of the pipeline")
    save_dir: PosixPath = Field(title="Save directory", description="Directory to save the sidecar")
    status: str = Field(title="Status", description="Status of the process")
    features: dict = Field(title="Key Features", description="Key features of the process", default={})
    input: dict = Field(title="Input", description="Input information")
    output: dict = Field(title="Output", description="Output information")
    processing: list[dict] = Field(title="Processing", description="Information about the processing from input to output")
    steps: list[str | dict] = Field(title="Steps", description="Steps in the process", default=[])
    metrics: list[dict[str, Any]] = Field(title="Metrics", description="Metrics of the process", default=[])
    error: Optional[str] = Field(title="Error", description="Error message", default=None)
    
    @field_validator("name", mode="after")
    def remove_extension(cls, name):
        name: str = name.split(".")[0]
        return name
    
    def save(self) -> None:
        """
        Save sidecar to file.
        """
        filepath: PosixPath = self.save_dir / f"{self.name}.json"
        sidecar: dict = self.model_dump()
        sidecar.pop("name")
        sidecar.pop("save_dir")
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(sidecar, f, indent=4)
            
    @classmethod
    def from_file(cls, filepath: str | PosixPath) -> "BIDSProcessSummarySidecar":
        """
        Load sidecar from file.
        """
        with open(filepath, "r") as f:
            data: dict = json.load(f)
        name: str = Path(filepath).name
        save_dir: PosixPath = Path(filepath).parent
        return cls(name=name, save_dir=save_dir, **data)

    @staticmethod
    def execute_process(function: Callable):
        """
        Decorator to execute a process and save the sidecar.
        The mandatory input arguments to the function should be
            input_filepath: str | PosixPath
            layout: BIDSLayout
            pipeline_name: str
            overwrite: bool
            
        The function should return a dictionary with the following keys:
            input: dict
            output: dict
            processing: list
            steps: list
            metrics: list
        """
        @functools.wraps(function)
        def execute(**kwargs) -> None:
            try:
                input_filepath: str = str(kwargs["input_filepath"])
                pipeline_name: str = kwargs["pipeline_name"]
                results: Optional[BIDSProcessResults] = function(**kwargs)
                if results:
                    input, output, processing, steps, metrics = results.input, results.output, results.processing, results.steps, results.metrics
                    process_id: Optional[str] = results.process_id
                    pipeline_id: Optional[str] = results.pipeline_id
                    output_filepath: str = output["path"]
                    save_dir: PosixPath = Path(output_filepath).parent
                    process_summary: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar(
                        process_id=process_id,
                        pipeline_id=pipeline_id,
                        name=Path(output_filepath).name,
                        pipeline_name=pipeline_name,
                        save_dir=save_dir,
                        status="success",
                        description=getattr(function, "__doc__", "Function description not available. Update the docstring."),
                        input=input,
                        output=output,
                        processing=processing,
                        steps=steps,
                        metrics=metrics
                    )
                    process_summary.save()
            except Exception as e:
                error_message: str = traceback.format_exc()
                input_filepath: str = str(kwargs["input_filepath"])
                print(f"Error processing {input_filepath} with {function.__name__}: {e}", "\n", error_message)
        return execute

class BIDSProcessResults(BaseModel):
    """
    Structure of the results of a BIDS Process.
    """
    process_id: Optional[str] = Field(description="Unique identifier for the process.", default=None)
    pipeline_id: Optional[str] = Field(description="Unique identifier for the pipeline.", default=None)
    input: dict[str, Any] = Field(description="Information about the input. Should always contain the key 'path'.")
    output: dict[str, Any] = Field(description="Information about the output. Should always contain the key 'path'.")
    processing: list[dict[str, Any]] = Field(description="Information about the processing steps.")
    steps: list[str] = Field(description="Steps taken during the process.")
    status: str = Field(description="Status of the process.")
    metrics: list[dict[str, Any]] = Field(description="Metrics generated by the process.")
    
    @field_validator("input", "output", mode="after")
    def check_input_output(cls, val: dict) -> dict:
        if "path" not in val:
            raise ValueError("Input and output must contain the key 'path'.")
        return val