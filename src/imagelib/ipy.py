from pathlib import Path, PosixPath

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, Dropdown, Output
import nibabel as nib
from bids import BIDSLayout
from IPython.display import display, clear_output

def explore_3D_array(arr: np.ndarray, cmap="gray", **kwargs):
    """
    Explore a 3D numpy array interactively.
    
    Args:
        arr: 3D numpy array
    """
    def fn(axis, slice_idx):
        plt.close("all")  # Close all existing figures
        plt.figure(figsize=(7, 7))
        if axis == 0:
            plt.imshow(arr[slice_idx, :, :], cmap=cmap, **kwargs)
        elif axis == 1:
            plt.imshow(arr[:, slice_idx, :], cmap=cmap, **kwargs)
        elif axis == 2:
            plt.imshow(arr[:, :, slice_idx], cmap=cmap, **kwargs)
        plt.colorbar()
        plt.title(f"Axis: {axis}, Slice: {slice_idx}")
        plt.axis("off")
        plt.show()

    interact(
        fn,
        axis=(0, 2),
        slice_idx=(0, arr.shape[0] - 1)
    )

def explore_4D_array(arr: np.ndarray, cmap="gray", **kwargs):
    """
    Explore a 4D numpy array interactively.
    
    Args:
        arr: 4D numpy array
    """
    def fn(axis, slice_idx, time_idx):
        plt.close("all")  # Close all existing figures
        plt.figure(figsize=(7, 7))
        if axis == 0:
            plt.imshow(arr[slice_idx, :, :, time_idx], cmap=cmap, **kwargs)
        elif axis == 1:
            plt.imshow(arr[:, slice_idx, :, time_idx], cmap=cmap, **kwargs)
        elif axis == 2:
            plt.imshow(arr[:, :, slice_idx, time_idx], cmap=cmap, **kwargs)
        plt.colorbar()
        plt.title(f"Axis: {axis}, Slice: {slice_idx}, Time: {time_idx}")
        plt.axis("off")
        plt.show()

    interact(
        fn,
        axis=(0, 2),
        slice_idx=(0, arr.shape[0] - 1),
        time_idx=(0, arr.shape[3] - 1)
    )

def explore_image(img: str | PosixPath | nib.Nifti1Image, cmap="gray", **kwargs):
    """
    Explore a 3D or 4D NIfTI image interactively.
    
    Args:
        img: Path to the image file or a Nifti1Image object.
    """
    if isinstance(img, str | PosixPath):
        img = nib.load(img)

    data = img.get_fdata()
    if data.ndim == 3:
        print("Exploring 3D data...")
        explore_3D_array(data, cmap=cmap, **kwargs)
    elif data.ndim == 4:
        print("Exploring 4D data...")
        explore_4D_array(data, cmap=cmap, **kwargs)
    else:
        raise ValueError("Only 3D and 4D data are supported.")
    
def explore_bids_directory(bids_dir):
    layout = BIDSLayout(bids_dir)
    
    # Get available subjects
    subjects = layout.get_subjects()
    subject_dropdown = widgets.Dropdown(options=subjects, description="Subject:")

    # Session Dropdown
    session_dropdown = widgets.Dropdown(description="Session:")
    
    # Image Dropdown
    image_dropdown = widgets.Dropdown(description="Image File:")
    
    # Function to update the sessions based on selected subject
    def update_sessions(subject: str):
        sessions = layout.get_sessions(subject=subject)
        session_dropdown.options = sessions
        session_dropdown.value = sessions[0]  # Default to the first session
    
    # Function to update the image files based on selected subject and session
    def update_images(subject: str, session: str):
        images = layout.get(subject=subject, session=session)
        img_files = [img.path for img in images if img.filename.endswith('.nii.gz')]
        image_dropdown.options = img_files
        image_dropdown.value = img_files[0] if img_files else None  # Default to the first image if available

    # Function to handle subject selection change
    def on_subject_change(change):
        subject = change.new
        update_sessions(subject)  # Update session options based on the selected subject
    
    # Function to handle session selection change
    def on_session_change(change):
        session = change.new
        subject = subject_dropdown.value
        update_images(subject, session)  # Update image options based on selected subject and session
    
    # Function to handle image selection change and print the file path
    def on_image_change(change):
        filename = change.new
        subject = subject_dropdown.value
        session = session_dropdown.value
        filepath = Path(bids_dir) / f"sub-{subject}" / f"ses-{session}" / filename
        # print(f"Selected file path: {filename}")
        explore_image(filename)

    display(subject_dropdown, session_dropdown, image_dropdown)

    # Attach event listeners
    subject_dropdown.observe(on_subject_change, names="value")
    session_dropdown.observe(on_session_change, names="value")
    image_dropdown.observe(on_image_change, names="value")
    
    # Initialize with the first subject's sessions and images
    update_sessions(subject_dropdown.value)
    update_images(subject_dropdown.value, session_dropdown.value)