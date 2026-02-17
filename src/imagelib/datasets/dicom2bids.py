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

from ..helpers.file import copy_as_symlinks, human_readable_size
from .dicom import Participant, Participants

import shutil, subprocess, json
from pathlib import Path, PosixPath
import multiprocessing, ast
from typing import Literal, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, FilePath, DirectoryPath, field_validator, Field
import pandas as pd
from rich.progress import track
    
class DICOMMigration(BaseModel):
    source_dir: DirectoryPath
    target_dir: DirectoryPath
    subject_id: str
    session_id: str
    participant_id: str
    bids_session_id: str
    symlink: bool = True

    @field_validator("subject_id", "participant_id", "session_id", "bids_session_id", mode="before")
    def coerce_ids(cls, values):
        values["subject_id"] = str(values["subject_id"])
        values["participant_id"] = str(values["participant_id"])
        values["session_id"] = str(values["session_id"])
        values["bids_session_id"] = str(values["bids_session_id"])
        return values
    
    @property
    def command(self) -> str:
        return f"cp -r {self.source_dir} {self.target_dir}" if not self.symlink else f"ln -s {self.source_dir} {self.target_dir}"

class NiftiMigration(DICOMMigration):
    filename: str
    comment: str
    bids_anon: bool = True
    gz_compress: bool = True
    
    @field_validator("subject_id", "participant_id", "session_id", "bids_session_id", mode="before")
    def coerce_ids(cls, value: str):
        return str(value)
    
    @field_validator("bids_anon", "gz_compress", mode="before")
    def coerce_bools(cls, value: str):
        return bool(value)
    
    @property
    def command(self) -> str:
        cmd: str = f"dcm2niix -z {'y' if self.gz_compress else 'n'} -f {self.filename} -o {self.target_dir} -w 0 -c '{self.comment}' "
        if self.bids_anon:
            cmd += "-ba y "
        else:
            cmd += "-ba n "
        cmd += str(self.source_dir)
        return "mkdir -p " + str(self.target_dir) + " && " + cmd

class DataMigrations(BaseModel):
    data_migrations: list[DICOMMigration]
    
    def __getitem__(self, index: int) -> DICOMMigration:
        return self.data_migrations[index]
    
    def filter(self, **kv_pairs) -> "DataMigrations":
        """
        Filter the data migrations.
        """
        data_migrations: list[DICOMMigration] = [migration for migration in self.data_migrations if all(getattr(migration, key) == value for key, value in kv_pairs.items())]
        return DataMigrations(data_migrations=data_migrations)
    
    def to_table(self, to_df: bool = False) -> list[dict] | pd.DataFrame:
        """
        Convert the data migrations to a table.
        """
        columns: list[str] = list(set(DICOMMigration.model_fields.keys()) | set(DICOMMigration.model_fields.keys()) | set(NiftiMigration.model_fields.keys()))
        table: list[dict] = []
        for migration in track(self.data_migrations, description="Converting data migrations to table"):
            row: dict = {column: None for column in columns}
            row.update(migration.model_dump())
            # Include command as a column
            row["command"] = migration.command
            table.append(row)
            
        return pd.DataFrame(table) if to_df else table
    
    @classmethod
    def from_table(cls, table: list[dict] | pd.DataFrame, sample: Optional[int] = None) -> "DataMigrations":
        """
        Create DataMigrations object from a table.
        
        Args:
            table: List of dictionaries or pandas DataFrame representing the data migrations.
            sample: If provided, only use the first 'sample' rows from the table.
            
        Returns:
            DataMigrations object.
        """
        if isinstance(table, pd.DataFrame):
            table = table.to_dict(orient="records")
        data_migrations: list[DICOMMigration] = []
        if sample is not None:
            table = table[:sample]
            print(f"Sampling first {sample} rows from the table to create DataMigrations object")
        for row in track(table, description="Creating DataMigrations object from table"):
            bids_anon: Optional[bool] = row.get("bids_anon", None)
            # Remove command from row if exists
            if "command" in row:
                row.pop("command")
            if bids_anon is None:
                migration: DICOMMigration = DICOMMigration(**row)
            else:
                migration: DICOMMigration = NiftiMigration(**row)
            data_migrations.append(migration)
        return DataMigrations(data_migrations=data_migrations)
    
    def execute_parallel(self) -> None:
        """
        Execute the data migrations.
        """
        def run_migration_cmd(cmd):
            try:
                subprocess.run(cmd, shell=True, check=True)
                return f"Command '{cmd}' completed successfully."
            except subprocess.CalledProcessError as e:
                return f"Command '{cmd}' failed with error: {e}"
            except Exception as e:
                return f"Command '{cmd}' failed with unexpected error: {e}"
            
        nifti_migration_cmds: list[str] = [migration.command for migration in self.data_migrations if isinstance(migration, NiftiMigration) and not (migration.target_dir / f"{migration.filename}.nii.gz").exists()]
        dicom_migration_cmds: list[DICOMMigration] = [migration for migration in self.data_migrations if isinstance(migration, DICOMMigration)]
            
        # Run the nifti conversions/migrations in parallel
        with ThreadPoolExecutor() as executor:
            # Get commands for Nifti migrations that are not yet completed
            
            
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
            
    def execute_sequential(self) -> None:
        """
        Execute the data migrations sequentially.
        """
        for migration in track(self.data_migrations, description="Executing data migrations"):
            cmd: str = migration.command
            subprocess.run(cmd, shell=True, check=False)

    def get_completion_status(self, sample: Optional[int] = None) -> pd.DataFrame:
        """
        Get the completion status of the data migrations.
        
        Args:
            sample: If provided, only check the first 'sample' migrations.
            
        Returns:
            pandas DataFrame with completion status.
        """
        if sample is not None:
            migrations: list[DICOMMigration] = self.data_migrations[:sample]
            print(f"Sampling first {len(migrations)} migrations for completion status")
        else:
            migrations: list[DICOMMigration] = self.data_migrations
            print(f"Checking completion status for all {len(migrations)} migrations")
        status_list: list[dict] = []
        for migration in track(migrations, description="Getting completion status of data migrations"):
            if isinstance(migration, NiftiMigration):
                expected_file: Path = Path(migration.target_dir) / f"{migration.filename}.nii.gz"
                completed: bool = expected_file.exists()
                # Get size of the file in readable format if completed
                file_size: int = expected_file.stat().st_size if completed else 0
                # Convert size to human-readable format
                file_size_hr = human_readable_size(file_size)
            else:
                expected_dir: Path = Path(migration.target_dir)
                completed: bool = expected_dir.exists() and any(expected_dir.iterdir())
                dir_size: int = sum(f.stat().st_size for f in expected_dir.rglob('*')) if completed else 0
                dir_size_hr = human_readable_size(dir_size)
            status_list.append({
                "participant_id": migration.participant_id,
                "subject_id": migration.subject_id,
                "bids_session_id": migration.bids_session_id,
                "session_id": migration.session_id,
                "completed": completed,
                "size": file_size_hr if isinstance(migration, NiftiMigration) else dir_size_hr
            })
        return pd.DataFrame(status_list)
    
    def get_completion_stats(self, df_status: pd.DataFrame, by: Literal["subject", "session", "all"] = "subject") -> pd.DataFrame:
        """
        Get completion statistics of the data migrations.
        
        Args:
            df_status: DataFrame with completion status.
            by: Grouping level for statistics - "subject", "session", or "all".
            
        Returns:
            pandas DataFrame with completion statistics.
        """
        if by == "subject":
            group_cols = ["participant_id"]
        elif by == "session":
            group_cols = ["participant_id", "bids_session_id"]
        else:
            group_cols = []
        
        if group_cols:
            stats: pd.DataFrame = df_status.groupby(group_cols).agg(
                total_migrations=pd.NamedAgg(column="completed", aggfunc="count"),
                completed_migrations=pd.NamedAgg(column="completed", aggfunc="sum")
            ).reset_index()
        else:
            total_migrations: int = len(df_status)
            completed_migrations: int = df_status["completed"].sum()
            stats: pd.DataFrame = pd.DataFrame([{
                "total_migrations": total_migrations,
                "completed_migrations": completed_migrations
            }])
        
        stats["completion_rate"] = stats["completed_migrations"] / stats["total_migrations"] * 100.0
        stats = stats.sort_values(by="completion_rate", ascending=False).reset_index(drop=True)
        return stats

class DICOMToBIDSConvertor(BaseModel):
    bids_root: DirectoryPath
    dicom_root: DirectoryPath
    participants: Participants
    config: Optional[FilePath] = None
    cmds: list[list[str]] = Field(default_factory=list)
    
    @field_validator("bids_root", mode="before")
    def create_bids_root(cls, value: str | Path):
        if not Path(value).exists():
            Path(value).mkdir(parents=True, exist_ok=True)
        return value
    
    def create_bids_scaffolding(self):
        """
        Create the scaffolding of the BIDS dataset.
        """
        # Create directory if not exists
        if not self.bids_root.exists():
            self.bids_root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["dcm2bids_scaffold", "-o", str(self.bids_root)], check=False)
        
    def prepare_migrations(self, symlink: bool = True, sample: bool = True, convert_to_nifti: bool = False, skip_missing: bool = True, save_to: Optional[Path] = None) -> DataMigrations:
        """
        Prepare the migrations for the DICOM dataset.
        """
        migrations: list[DICOMMigration] = []
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
                    for study in session_info.studies:
                        dicom_dir: Path = Path(study.study_subdir)
                        if not dicom_dir.exists() and skip_missing:
                            print(f"Skipping {dicom_dir} as it does not exist")
                            continue
                        
                        migration: NiftiMigration = NiftiMigration(
                            source_dir=dicom_dir,
                            target_dir=session_dir,
                            subject_id=participant.subject_id,
                            session_id=session_info.session_id,
                            participant_id=participant.participant_id,
                            bids_session_id=session_info.bids_session_id,
                            symlink=symlink,
                            filename=study.study_name,
                            comment=f"Converted from DICOM study {study.study_name}",
                            bids_anon=True
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
                
        if save_to:
            data_migrations: DataMigrations = DataMigrations(data_migrations=migrations)
            table: pd.DataFrame = data_migrations.to_table(to_df=True)
            table.to_csv(save_to, index=False)
            print(f"Saved data migrations to {save_to}")
            
        return DataMigrations(data_migrations=migrations)
            
    def run_dcm2bids_helper(self, participant_id: str, bids_session_id: str, output_dir: PosixPath = Path("tmp/")) -> None:
        """
        Run dcm2bids_helper to create example sidecar json files.
        """
        participant: Participant = self.participants.filter(participant_ids=[participant_id], bids_session_ids=[bids_session_id]).participants[0]
        dicom_subdir: str = participant.sessions[0].dicom_subdir
        dicom_subdir_full_path: PosixPath = self.dicom_root / dicom_subdir
        subprocess.run(["dcm2bids_helper", "-d", str(dicom_subdir_full_path), "-o", str(output_dir)], check=True)

    def generate_cmds(self, data_migrations: str | Path | DataMigrations, skip_dcm2niix: bool = False, auto_extract_entities: bool = True) -> None:
        """
        Generate dcm2bids commands for all participants. If skip_dcm2niix is True, skip dcm2niix conversion.
        """
        if isinstance(data_migrations, (str, Path)):
            data_migrations: DataMigrations = DataMigrations.from_table(pd.read_csv(data_migrations))
        
        if not self.config:
            raise ValueError("Configuration file not provided")
        
        # Running dcm2bids on session-level data, but migrations data is at study-level
        df_migrations: pd.DataFrame = data_migrations.to_table(to_df=True)
        # Remove duplicate participant_id and bids_session_id combinations
        df_sessions: pd.DataFrame = df_migrations[["participant_id", "bids_session_id", "target_dir"]].drop_duplicates()
        for _, row in df_sessions.iterrows():
            participant_id: str = row["participant_id"]
            bids_session_id: str = row["bids_session_id"]
            data_dir: Path = Path(row["target_dir"])
            command: list[str] = ["dcm2bids", "-d", str(data_dir), "-p", participant_id, "-s", bids_session_id, "-c", str(self.config), "-o", str(self.bids_root)]
            if skip_dcm2niix:
                command.append("--skip_dcm2niix")
            if auto_extract_entities:
                command.append("--auto_extract_entities")
            self.cmds.append(command)
        
        # for migration in data_migrations.data_migrations:
        #     participant_id: str = migration.participant_id
        #     bids_session_id: str = migration.bids_session_id
        #     data_dir: Path = migration.target_dir
        #     command: list[str] = ["dcm2bids", "-d", str(data_dir), "-p", participant_id, "-s", bids_session_id, "-c", str(self.config), "-o", str(self.bids_root)]
        #     if skip_dcm2niix:
        #         command.append("--skip_dcm2niix")
        #     if auto_extract_entities:
        #         command.append("--auto_extract_entities")
        #     self.cmds.append(command)

    def save_cmds(self, output_file: str | PosixPath) -> None:
        """
        Save the dcm2bids commands to a file.
        """
        output_file = Path(output_file)
        with open(output_file, "w") as f:
            for cmd in self.cmds:
                f.write(" ".join(cmd) + "\n")
        print(f"Saved dcm2bids commands to {output_file}")
        
    def convert2bids_parallel(self) -> None:
        """
        Convert DICOM data to BIDS format for all participants in parallel.
        """
        with multiprocessing.Pool() as pool:
            pool.map(subprocess.run, self.cmds)
            
    def convert2bids_sequential(self) -> None:
        """
        Convert DICOM data to BIDS format for all participants sequentially.
        """
        for cmd in track(self.cmds, description="Converting DICOM data to BIDS format"):
            subprocess.run(cmd, check=False)
