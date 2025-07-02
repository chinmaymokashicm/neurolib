"""
Work with NIfTI files.
"""
from pathlib import Path, PosixPath
from typing import Optional
import os, shutil

def flatten_nifti_structure(nifti_root: PosixPath, output_dir: PosixPath, prepare_only: bool = True) -> dict:
    """
    Flatten the NIfTI file structure by moving all files to a single directory.
    Assumptions:
    - The NIfTI files are organized in the following structure:
        <nifti_root>/
            ├── <subject_id>/
            │   ├── <session_id>/
            │   │   ├── <scan_id>/
            │   │   │   ├── <file1>
            │   │   │   └── <file2>
            │   │   └── <scan_id2>/
            │   └── <session_id2>/
            └── <subject_id2>/
    - Subject IDs are unique. Session IDs append to the subject ID. Example: 1801-1001-3912-9329
    Args:
        nifti_root (PosixPath): The root directory containing the NIfTI files.
        output_dir (PosixPath): The output directory where the flattened files will be saved.
        prepare_only (bool): If True, only prepare the output directory without copying files.
    Returns:
        dict: A dictionary representing the directory tree of the NIfTI files.
    """
    nifti_root: PosixPath = Path(nifti_root)
    output_dir: PosixPath = Path(output_dir)
    if not nifti_root.exists():
        raise FileNotFoundError(f"NIfTI root directory {nifti_root} does not exist.")
    if not nifti_root.is_dir():
        raise NotADirectoryError(f"NIfTI root {nifti_root} is not a directory.")
    
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        
    dir_tree: dict = {}
        
    for subject_dir in nifti_root.iterdir():
        if not subject_dir.is_dir():
            continue
        dir_tree[subject_dir.name] = {}
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue
            relevant_file_extensions = [".nii", ".nii.gz", ".json", ".bval", ".bvec"]
            relevant_files = [f for f in session_dir.glob("*") if any(f.name.endswith(ext) for ext in relevant_file_extensions)]
            if not relevant_files:
                continue
            dir_tree[subject_dir.name][session_dir.name] = relevant_files
    
    if prepare_only:
        return dir_tree
    for subject_id, sessions in dir_tree.items():
        for session_id, files in sessions.items():
            for file in files:
                if file.is_file():
                    new_file_name = f"{subject_id}_{session_id}_{file.name}"
                    new_file_path = output_dir / new_file_name
                    if not new_file_path.exists():
                        shutil.copy(file, new_file_path)
                        print(f"Copied {file} to {new_file_path}")
    return dir_tree