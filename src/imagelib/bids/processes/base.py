from ...helpers.generate import generate_id

from typing import Optional, Any, Callable, Protocol, runtime_checkable
from pathlib import Path, PosixPath
import functools, traceback, json, os
from copy import deepcopy

from pydantic import BaseModel, Field, field_validator
from bids import BIDSLayout
from rich.progress import track

@runtime_checkable
class BIDSProcessLogicCallable(Protocol):
    def __call__(
        self,
        input_filepath: str | PosixPath,
        layout: BIDSLayout,
        pipeline_name: str,
        overwrite: bool = False,
        process_id: Optional[str] = None,
        process_exec_id: Optional[str] = None,
        pipeline_id: Optional[str] = None,
    ) -> None:
        """
        Process the input file using the provided arguments.
        Args:
            input_filepath (str | PosixPath): Path to the input file.
            layout (BIDSLayout): BIDS layout object.
            pipeline_name (str): Name of the pipeline.
            overwrite (bool): Whether to overwrite existing outputs. Default is False.
            process_id (Optional[str]): ID of the process. Default is None.
            process_exec_id (Optional[str]): Execution ID of the process. Default is None.
            pipeline_id (Optional[str]): ID of the pipeline. Default is None.
        Returns:
            None
        """
        ...

class BIDSProcess(BaseModel):
    """
    Define a unitary process in the BIDS pipeline.
    """
    id: str = Field(title="ID. The unique identifier of the process image.", default_factory=lambda: generate_id("PR", 10, "-"))
    name: str = Field(title="Name", description="Name of the process")
    description: Optional[str] = Field(title="Description", description="Description of the process", default=None)
    logic: Callable = Field(title="Logic", description="Logic of the process. A callable object.")
    
    @field_validator("logic", mode="after")
    def check_logic(cls, val: Callable) -> Callable:
        if not callable(val):
            raise ValueError("Logic must be a callable object.")
        if not isinstance(val, BIDSProcessLogicCallable):
            raise ValueError("Logic must be a callable object with the correct signature.")
        return val
    
    def to_dict(self) -> dict:
        """
        Convert the process to a dictionary.
        """
        return {
            "name": self.name,
            "description": self.description,
            "logic": getattr(self.logic, "__doc__", "Function description not available. Update the docstring."),
        }
    
    def set_id_from_env(self):
        """
        Set the process ID from environment variables.
        """
        self.id = os.getenv("PROCESS_ID", self.id)

class BIDSProcessExec(BaseModel):
    """
    Define a process execution in the BIDS pipeline.
    Add I/O metadata to the BIDSProcess.
    """
    id: str = Field(title="ID. The unique identifier of the process execution.", default_factory=lambda: generate_id("PE", 10, "-"))
    process: BIDSProcess = Field(title="Process", description="Process to be executed")
    bids_roots: list[PosixPath] = Field(title="BIDS roots", description="List of BIDS root directories")
    bids_filters: dict = Field(title="BIDS filters", description="BIDS filters to apply to the input files", default_factory=dict)
    pipeline_name: Optional[str] = Field(title="Pipeline name", description="Name of the pipeline", default=None)
    overwrite: bool = Field(title="Overwrite", description="Overwrite existing files", default=False)
    pipeline_id: Optional[str] = Field(title="Pipeline ID", description="Unique identifier for the pipeline", default=None)
    extra_kwargs: dict = Field(title="Extra keyword arguments", description="Extra keyword arguments for the process", default={})
    
    @field_validator("bids_roots", mode="after")
    def check_bids_roots(cls, val: list[PosixPath]) -> list[PosixPath]:
        if not val:
            raise ValueError("BIDS roots cannot be empty.")
        for root in val:
            if not isinstance(root, PosixPath):
                raise ValueError("BIDS roots must be PosixPath objects.")
            BIDSLayout(root, derivatives=True)
        return val
    
    @classmethod
    def quick_create(cls, logic: Callable, bids_roots: list[PosixPath], bids_filters: Optional[dict] = {}, extra_kwargs: Optional[dict] = None) -> "BIDSProcessExec":
        """
        Quick create a BIDSProcessExec object. Purpose is to create a process execution object with low redundancy and high readability.
        """
        if bids_filters is None:
            bids_filters = {}
        if extra_kwargs is None:
            extra_kwargs = {}
        process: BIDSProcess = BIDSProcess(
            name=logic.__name__,
            description=getattr(logic, "__doc__", "Function description not available. Update the docstring."),
            logic=logic,
        )
        return cls(
            process=process,
            bids_roots=bids_roots,
            bids_filters=bids_filters,
            extra_kwargs=extra_kwargs
        )
    
    def set_values_from_env(self):
        """
        Set the pipeline name and overwrite values from environment variables.
        """
        self.id = os.getenv("PROCESS_EXEC_ID", self.id)
        if os.environ("BIDS_FILTERS") is not None:
            self.bids_filters = json.loads(os.getenv("BIDS_FILTERS"))
    
    def get_layout_filepaths(self) -> dict[str, list[str]]:
        """
        Get the filepaths for each BIDS layout for the process execution.
        """
        layout_filepaths: dict[str, list[str]] = {}
        for root in self.bids_roots:
            layout: BIDSLayout = BIDSLayout(root, derivatives=True)
            layout_filepaths[root] = layout.get(return_type="file", **self.bids_filters)
        return layout_filepaths
    
    def __get_execution_plan(self, pipeline_id: Optional[str] = None) -> dict[str, list[str]]:
        """
        Get the execution plan for the process execution.
        """
        layout_filepaths: dict[str, list[str]] = self.get_layout_filepaths()
        execution_plan: dict = {}
        for root, filepaths in layout_filepaths.items():
            execution_plan[root] = {}
            for filepath in filepaths:
                execution_plan[root][filepath] = {
                    "input_filepath": filepath,
                    "layout": BIDSLayout(root, derivatives=True),
                    "pipeline_name": self.pipeline_name,
                    "overwrite": self.overwrite,
                    "process_id": self.process.id,
                    "process_exec_id": self.id,
                    "pipeline_id": pipeline_id if pipeline_id is not None else self.pipeline_id,
                }
                execution_plan[root][filepath].update(self.extra_kwargs)
        return execution_plan
    
    def execute(self) -> None:
        """
        Execute the process.
        """
        # Check if pipeline_name is set
        if self.pipeline_name is None:
            raise ValueError("Pipeline name must be set before executing the process.")
        
        execution_plan: dict[str, list[str]] = self.__get_execution_plan()
        for root_dir, filepath_kwargs in execution_plan.items():
            for filepath, kwargs in track(filepath_kwargs.items(), description=f"Processing {self.process.name} on {len(filepath_kwargs)} files on {root_dir}"):
                try:
                    self.process.logic(**kwargs)
                except Exception as e:
                    error_message: str = traceback.format_exc()
                    print(f"Error processing {filepath} with {self.process.name}: {e}", "\n", error_message)
                    
    def to_dict(self) -> dict:
        """
        Convert the process execution to a dictionary.
        """
        return {
            "id": self.id,
            "process": self.process.to_dict(),
            "bids_roots": [str(root) for root in self.bids_roots],
            "bids_filters": self.bids_filters,
            "pipeline_name": self.pipeline_name,
            "overwrite": self.overwrite,
            "pipeline_id": self.pipeline_id,
            "extra_kwargs": self.extra_kwargs,
        }


# class BIDSProcess(BaseModel):
#     """
#     Define a process in the BIDS pipeline.
#     """
#     name: str = Field(title="Name", description="Name of the process")
#     description: Optional[str] = Field(title="Description", description="Description of the process", default=None)
#     logic: Callable = Field(title="Logic", description="Logic of the process. A callable object.")
#     bids_filters: dict = Field(title="BIDS filters", description="BIDS filters to apply to the input files", default={})
#     kwargs: dict = Field(title="Keyword arguments", description="Keyword arguments for the process", default={})
    
#     def execute(self) -> Optional[str]:
#         """
#         Execute the process.
#         """
#         return self.logic(**self.kwargs)
    
#     def to_dict(self) -> dict:
#         """
#         Convert the process to a dictionary.
#         """
#         kwargs: dict = {}
#         for key, value in self.kwargs.items():
#             if isinstance(value, PosixPath):
#                 value = str(value)
#             elif isinstance(value, BIDSLayout):
#                 value = value.root
#             kwargs[key] = value
#         return {
#             "name": self.name,
#             "description": self.description,
#             "kwargs": kwargs
#         }
    
#     def set_input_filepaths(self, layout: BIDSLayout, bids_filters: dict):
#         """
#         Set the input filepaths for the process.
#         """
#         self.input_filepaths = layout.get(return_type="file", **bids_filters)

class BIDSProcessSummarySidecar(BaseModel):
    """
    Sidecars that summarize the process.
    """
    process_id: Optional[str] = Field(title="Process ID", description="Unique identifier for the process", default=None)
    process_exec_id: Optional[str] = Field(title="Process Execution ID", description="Unique identifier for the process execution", default=None)
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
                    process_exec_id: Optional[str] = results.process_exec_id
                    pipeline_id: Optional[str] = results.pipeline_id
                    output_filepath: str = output["path"]
                    save_dir: PosixPath = Path(output_filepath).parent
                    process_summary: BIDSProcessSummarySidecar = BIDSProcessSummarySidecar(
                        process_id=process_id,
                        process_exec_id=process_exec_id,
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
    process_exec_id: Optional[str] = Field(description="Unique identifier for the process execution.", default=None)
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