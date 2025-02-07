import os

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