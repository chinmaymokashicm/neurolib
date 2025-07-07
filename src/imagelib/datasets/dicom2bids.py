"""
Convert DICOM dataset to BIDS dataset. The organization of the DICOM dataset is variable.
Use of the tool dcm2bids as the package for the conversion.

Steps of conversion-
0. Installation of dcm2bids and dcm2niix
1. Create a scaffolding of the BIDS dataset
2. Create a 1:1 mapping of existing subject ID and session ID to BIDS participant ID and session ID (to be done by the user - not here)
2. Migrate the DICOM dataset to sourcedata/ subdirectory (per subject and session)
3. Use dcm2bids_helper to create example sidecar json files (optional)
3. Build the configuration file for dcm2bids - use from sidecar json files (non-programmatically)
4. Run dcm2bids with each session of the DICOM dataset having a unique participant ID and session ID
"""

from ..helpers.file import copy_as_symlinks
from .dicom import Participant, Participants

import shutil, subprocess, json
from pathlib import Path, PosixPath
import multiprocessing, ast
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, FilePath, DirectoryPath, field_validator, Field
import pandas as pd
from rich.progress import track
    
class DataMigration(BaseModel):
    source_dir: DirectoryPath
    target_dir: DirectoryPath
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    participant_id: Optional[str] = None
    bids_session_id: Optional[str] = None

class DICOMMigration(DataMigration):
    symlink: bool = True
    
    @field_validator("subject_id", "participant_id", "session_id", "bids_session_id", mode="before")
    def coerce_ids(cls, values):
        values["subject_id"] = str(values["subject_id"])
        values["participant_id"] = str(values["participant_id"])
        values["session_id"] = str(values["session_id"])
        values["bids_session_id"] = str(values["bids_session_id"])
        return values
    
    @field_validator("symlink", mode="before")
    def coerce_symlink(cls, value: str):
        return bool(value)

class NiftiMigration(DataMigration):
    filename: str
    comment: Optional[str] = None
    bids_anon: bool = True
    gz_compress: bool = True
    cmd: list[str] = Field(default_factory=list)
    
    @field_validator("subject_id", "participant_id", "session_id", "bids_session_id", mode="before")
    def coerce_ids(cls, value: str):
        return str(value)
    
    @field_validator("bids_anon", "gz_compress", mode="before")
    def coerce_bools(cls, value: str):
        return bool(value)
    
    @field_validator("cmd", mode="before")
    def coerce_cmd(cls, value: list[str] | str):
        return ast.literal_eval(value) if isinstance(value, str) else value

class DataMigrations(BaseModel):
    data_migrations: list[DataMigration]
    
    def __getitem__(self, index: int) -> DataMigration:
        return self.data_migrations[index]
    
    def filter(self, **kv_pairs) -> "DataMigrations":
        """
        Filter the data migrations.
        """
        data_migrations: list[DataMigration] = [migration for migration in self.data_migrations if all(getattr(migration, key) == value for key, value in kv_pairs.items())]
        return DataMigrations(data_migrations=data_migrations)
    
    def to_table(self, to_df: bool = False) -> list[dict] | pd.DataFrame:
        """
        Convert the data migrations to a table.
        """
        columns: list[str] = list(set(DataMigration.model_fields.keys()) | set(DICOMMigration.model_fields.keys()) | set(NiftiMigration.model_fields.keys()))
        table: list[dict] = []
        for migration in track(self.data_migrations, description="Converting data migrations to table"):
            row: dict = {column: None for column in columns}
            row.update(migration.model_dump())
            table.append(row)
            
        return pd.DataFrame(table) if to_df else table
    
    @classmethod
    def from_table(cls, table: list[dict] | pd.DataFrame) -> "DataMigrations":
        """
        Create DataMigrations object from a table.
        """
        if isinstance(table, pd.DataFrame):
            table = table.to_dict(orient="records")
        data_migrations: list[DataMigration] = []
        for row in track(table, description="Creating DataMigrations object from table"):
            bids_anon: Optional[bool] = row.get("bids_anon", None)
            if bids_anon is None:
                migration: DataMigration = DICOMMigration(**row)
            else:
                migration: DataMigration = NiftiMigration(**row)
            data_migrations.append(migration)
        return DataMigrations(data_migrations=data_migrations)
    
    def execute(self) -> None:
        """
        Execute the data migrations.
        """
        def run_migration_cmd(cmd):
            try:
                subprocess.run(cmd, check=True)
                return f"Command {cmd} completed successfully."
            except subprocess.CalledProcessError as e:
                return f"Command {cmd} failed with error: {e}"
            
        nifti_migration_cmds: list[list[str]] = [migration.cmd for migration in self.data_migrations if isinstance(migration, NiftiMigration) and not (migration.target_dir / f"{migration.filename}.nii.gz").exists()]
        dicom_migration_cmds: list[DataMigration] = [migration for migration in self.data_migrations if isinstance(migration, DICOMMigration)]
            
        # Run the nifti conversions/migrations in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(run_migration_cmd, cmd) for cmd in track(nifti_migration_cmds, description="Executing Nifti migrations")]
            for future in as_completed(futures):
                print(future.result())
        
        # for migration in track(self.data_migrations, description="Executing DICOM migrations"):
        for migration in track(dicom_migration_cmds, description="Executing DICOM migrations"):
            if isinstance(migration, DICOMMigration):
                if migration.symlink:
                    copy_as_symlinks(migration.source_dir, migration.target_dir)
                else:
                    shutil.copytree(migration.source_dir, migration.target_dir, dirs_exist_ok=True)
            else:
                raise ValueError(f"Unsupported migration type: {type(migration)}")

class NiftiData(BaseModel):
    data_migrations: DataMigrations
    
    def aggregate_field_values(self, fields: str | list[str], sample: Optional[int] = None, verbose: bool = True) -> dict:
        """
        Aggregate field values from the data migrations.
        """
        if not all(isinstance(migration, NiftiMigration) for migration in self.data_migrations.data_migrations):
            raise ValueError("Data migrations contain non-Nifti migrations")
        fields: list[str] = [fields] if isinstance(fields, str) else fields
        aggregated_data: dict = {field: {} for field in fields}
        data_migrations: list[NiftiMigration] = self.data_migrations.data_migrations[:sample] if sample else self.data_migrations.data_migrations
        for data_migration in track(data_migrations, description=f"Aggregating field values for {', '.join(fields)} from {len(data_migrations)} data migrations"):
            filepath: PosixPath = Path(data_migration.target_dir) / f"{data_migration.filename}.json"
            if not filepath.exists():
                if verbose:
                    print(f"File {filepath} does not exist")
                continue
            with open(filepath, "r") as f:
                sidecar_data: dict = json.load(f)
            for field in fields:
                value = sidecar_data.get(field, None)
                if isinstance(value, list):
                    raise ValueError(f"Field {field} is a list")
                if value not in aggregated_data[field]:
                    aggregated_data[field][value] = 0
                aggregated_data[field][value] += 1
                # print(f"Field {field}: {value}")
        return {field: dict(sorted(values.items(), key=lambda item: item[1], reverse=True)) for field, values in aggregated_data.items()}

class DICOMToBIDSConvertor(BaseModel):
    bids_root: DirectoryPath
    dicom_root: DirectoryPath
    participants: Participants
    config: Optional[FilePath] = None
    cmds: list[list[str]] = Field(default_factory=list)
    
    @field_validator("bids_root", mode="before")
    def create_bids_root(cls, value):
        if not value.exists():
            value.mkdir(parents=True, exist_ok=True)
        return value
    
    def create_bids_scaffolding(self):
        """
        Create the scaffolding of the BIDS dataset.
        """
        # Create directory if not exists
        if not self.bids_root.exists():
            self.bids_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dcm2bids_scaffold", "-o", str(self.bids_root)], check=False)
        
    def prepare_migrations(self, symlink: bool = True, sample: bool = True, convert_to_nifti: bool = False, skip_missing: bool = True) -> DataMigrations:
        """
        Prepare the migrations for the DICOM dataset.
        """
        migrations: list[DataMigration] = []
        end: int = 2 if sample else len(self.participants.participants)
        
        for participant in self.participants.participants[:end]:
            # Create participant and session directories
            participant_dir = self.bids_root / "sourcedata" / participant.subject_id
            if not participant_dir.exists():
                participant_dir.mkdir(parents=True, exist_ok=True)
            
            for session_info in participant.sessions:
                session_dir = participant_dir / session_info.session_id
                if not session_dir.exists():
                    session_dir.mkdir(parents=True, exist_ok=True)
                
                if convert_to_nifti:
                    # Convert DICOM to NIFTI
                    for scan in session_info.scans:
                        dicom_dir: PosixPath = self.dicom_root / session_info.dicom_subdir / scan.series_name
                        if not dicom_dir.exists() and skip_missing:
                            print(f"Skipping {dicom_dir} as it does not exist")
                            continue
                        filename: str = f"{scan.series_name}"
                        comment: str = f"Sourced from {dicom_dir}"
                        cmd: list[str] = ["dcm2niix", "-z", "y", "-o", str(session_dir), "-f", filename, "-ba", "y", "-c", comment, str(dicom_dir)]
                        
                        migration: NiftiMigration = NiftiMigration(
                            source_dir=dicom_dir,
                            target_dir=session_dir,
                            subject_id=participant.subject_id,
                            session_id=session_info.session_id,
                            participant_id=participant.participant_id,
                            bids_session_id=session_info.bids_session_id,
                            filename=filename,
                            comment=comment,
                            cmd=cmd
                        )
                        
                        migrations.append(migration)
                    continue
                
                # Migrate DICOM data to sourcedata/ subdirectory
                dicom_dir = self.dicom_root / session_info.dicom_subdir
                migration: DICOMMigration = DICOMMigration(
                    source_dir=dicom_dir,
                    target_dir=session_dir,
                    subject_id=participant.subject_id,
                    session_id=session_info.session_id,
                    participant_id=participant.participant_id,
                    bids_session_id=session_info.bids_session_id,
                    symlink=symlink
                )
                migrations.append(migration)
                
        return DataMigrations(data_migrations=migrations)
            
    def run_dcm2bids_helper(self, participant_id: str, bids_session_id: str, output_dir: PosixPath = Path("tmp/")) -> None:
        """
        Run dcm2bids_helper to create example sidecar json files.
        """
        participant: Participant = self.participants.filter(participant_ids=[participant_id], bids_session_ids=[bids_session_id]).participants[0]
        dicom_subdir: str = participant.sessions[0].dicom_subdir
        dicom_subdir_full_path: PosixPath = self.dicom_root / dicom_subdir
        subprocess.run(["dcm2bids_helper", "-d", str(dicom_subdir_full_path), "-o", str(output_dir)], check=True)

    def generate_cmds(self, nifti2bids: bool = False, auto_extract_entities: bool = True) -> None:
        """
        Generate dcm2bids commands for all participants. If nifti2bids is True, skip dcm2niix conversion.
        """
        if not self.config:
            raise ValueError("Configuration file not provided")
        if nifti2bids:
            # Skip dcm2niix conversion. 
            # Instead of reading from the DICOM directory, read from the NIFTI directory (bids_root/sourcedata)
            for participant in self.participants.participants:
                for session_info in participant.sessions:
                    nifti_dir: PosixPath = self.bids_root / "sourcedata" / participant.subject_id / session_info.session_id
                    cmd: list[str] = ["dcm2bids", "-d", str(nifti_dir), "-p", participant.participant_id, "-s", session_info.bids_session_id, "-c", str(self.config), "-o", str(self.bids_root), "--skip_dcm2niix"]
                    if auto_extract_entities:
                        cmd.append("--auto_extract_entities")
                    self.cmds.append(cmd)
        else:
            for participant in self.participants.participants:
                for session_info in participant.sessions:
                    dicom_subdir: str = session_info.dicom_subdir
                    cmd: list[str] = ["dcm2bids", "-d", dicom_subdir, "-p", participant.participant_id, "-s", session_info.bids_session_id, "-c", str(self.config), "-o", str(self.bids_root)]
                    if auto_extract_entities:
                        cmd.append("--auto_extract_entities")
                    self.cmds.append(cmd)
                
    def save_cmds(self, output_file: str | PosixPath) -> None:
        """
        Save the dcm2bids commands to a file.
        """
        output_file = Path(output_file)
        with open(output_file, "w") as f:
            for cmd in self.cmds:
                f.write(" ".join(cmd) + "\n")
        print(f"Saved dcm2bids commands to {output_file}")
        
    def convert2bids(self) -> None:
        """
        Convert DICOM data to BIDS format for all participants.
        """
        # with multiprocessing.Pool() as pool:
        #     pool.starmap(self.convert2bids_per_participant, [(participant.participant_id, session_info.bids_session_id) for participant in self.participants for session_info in participant.sessions])
        with multiprocessing.Pool() as pool:
            pool.map(subprocess.run, self.cmds)
