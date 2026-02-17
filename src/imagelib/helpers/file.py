import os, yaml, zipfile
from pathlib import Path
from typing import Optional

def copy_as_symlinks(src, dst):
    """
    Copy files from src to dst as symlinks.
    
    Args:
        src (str): Source directory
        dst (str): Destination directory
    """
    if not os.path.exists(dst):
        os.makedirs(dst)

    for root, dirs, files in os.walk(src):
        rel_path = os.path.relpath(root, src)
        dest_root = os.path.join(dst, rel_path) if rel_path != "." else dst

        # Ensure the destination directory exists
        os.makedirs(dest_root, exist_ok=True)

        # Create symlinks for files
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dest_root, file)
            
            if not os.path.exists(dst_file):  # Avoid overwriting
                os.symlink(src_file, dst_file)

        # Create empty directories
        for dir in dirs:
            dst_dir = os.path.join(dest_root, dir)
            os.makedirs(dst_dir, exist_ok=True)
            
def generate_zip_file_tree(zip_file_path: str, yaml_filepath: Optional[str] = None) -> dict:
    """
    Get the directory tree of a zip file.
    
    Args:
        zip_file_path (str): Path to the zip file.
        yaml_filepath (Optional[str]): If provided, save the directory tree as a YAML file at this path.
        
    Returns:
        dict: A dictionary representing the directory tree of the zip file.
    """
    zip_file_path: Path = Path(zip_file_path)
    if not str(zip_file_path).endswith('.zip'):
        raise ValueError("Provided file is not a zip file. Please provide a valid zip file path.")
    
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        dir_tree = {}
        for file_info in zip_ref.infolist():
            parts = file_info.filename.split("/")
            # Skip empty parts (can happen for directories)
            parts = [p for p in parts if p]
            if not parts:
                continue
            current_level = dir_tree
            # Only process directories, skip files
            if not file_info.is_dir():
                # Only go up to the parent directory, skip the filename
                parts = parts[:-1]
            for part in parts:
                if part not in current_level:
                    current_level[part] = {}
                current_level = current_level[part]
    
    if yaml_filepath:
        with open(yaml_filepath, "w") as yaml_file:
            yaml.dump(dir_tree, yaml_file, default_flow_style=False)
        print(f"Directory tree saved as YAML at {yaml_filepath}")
    
    return dir_tree

def merge_zip_file_trees(zip_trees: list[dict], save_as_yaml: bool = False) -> dict:
    """
    Merge multiple zip file directory trees into one.
    
    Args:
        zip_trees (list[dict]): List of directory trees from zip files.
        save_as_yaml (bool): If True, save the merged directory tree as a YAML file.
        
    Returns:
        dict: A merged dictionary representing the directory tree.
    """
    merged_tree = {}
    
    for tree in zip_trees:
        def merge_dicts(source, target):
            for key, value in source.items():
                if isinstance(value, dict):
                    target[key] = target.get(key, {})
                    merge_dicts(value, target[key])
                else:
                    target[key] = value
        merge_dicts(tree, merged_tree)
    
    if save_as_yaml:
        yaml_file_path = "merged_zip_tree.yaml"
        with open(yaml_file_path, 'w') as yaml_file:
            yaml.dump(merged_tree, yaml_file, default_flow_style=False)
        print(f"Merged directory tree saved as YAML at {yaml_file_path}")
    
    return merged_tree

def human_readable_size(size, decimal_places=2):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024.0
    return f"{size:.{decimal_places}f} PB"