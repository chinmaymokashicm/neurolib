import ipywidgets as widgets
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from bids import BIDSLayout
from ipywidgets import interact, Output
from IPython.display import display

def explore_3D_array(arr: np.ndarray, output_widget: Output, cmap="gray", **kwargs):
    """
    Explore a 3D numpy array interactively.
    
    Args:
        arr: 3D numpy array
        output_widget: Output widget to control rendering.
    """
    def fn(axis, slice_idx):
        with output_widget:
            output_widget.clear_output(wait=True)  # Clear previous content
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

def explore_4D_array(arr: np.ndarray, output_widget: Output, cmap="gray", **kwargs):
    """
    Explore a 4D numpy array interactively.
    
    Args:
        arr: 4D numpy array
        output_widget: Output widget to control rendering.
    """
    def fn(axis, slice_idx, time_idx):
        with output_widget:
            output_widget.clear_output(wait=True)  # Clear previous content
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

def explore_image(img: str | nib.Nifti1Image, output_widget: Output, cmap="gray", **kwargs):
    """
    Explore a 3D or 4D NIfTI image interactively.
    
    Args:
        img: Path to the image file or a Nifti1Image object.
        output_widget: Output widget to control rendering.
    """
    if isinstance(img, str):
        img = nib.load(img)

    data = img.get_fdata()
    output_widget.clear_output(wait=True)  # Clear previous output immediately
    if data.ndim == 3:
        explore_3D_array(data, output_widget, cmap=cmap, **kwargs)
    elif data.ndim == 4:
        explore_4D_array(data, output_widget, cmap=cmap, **kwargs)
    else:
        raise ValueError("Only 3D and 4D data are supported.")

def explore_bids_directory(bids_dir):
    layout = BIDSLayout(bids_dir)
    
    # Widgets
    subject_dropdown = widgets.Dropdown(description="Subject:")
    session_dropdown = widgets.Dropdown(description="Session:")
    image_dropdown = widgets.Dropdown(description="Image File:")
    output_widget = Output()  # Widget to manage plots
    
    # Update session options based on subject
    def update_sessions(subject):
        sessions = layout.get_sessions(subject=subject)
        session_dropdown.options = sessions
        if sessions:
            session_dropdown.value = sessions[0]
    
    # Update image file options based on subject and session
    def update_images(subject, session):
        images = layout.get(subject=subject, session=session, suffix="bold")
        img_files = [img.path for img in images if img.path.endswith('.nii.gz')]
        image_dropdown.options = img_files
        if img_files:
            image_dropdown.value = img_files[0]

    # Handle subject selection
    def on_subject_change(change):
        subject = change.new
        update_sessions(subject)
        update_images(subject, session_dropdown.value)

    # Handle session selection
    def on_session_change(change):
        session = change.new
        update_images(subject_dropdown.value, session)

    # Handle image selection and display
    def on_image_change(change):
        file_path = change.new
        if file_path:
            with output_widget:
                print(f"Displaying image: {file_path}")
                explore_image(file_path, output_widget)

    # Attach event listeners
    subject_dropdown.observe(on_subject_change, names="value")
    session_dropdown.observe(on_session_change, names="value")
    image_dropdown.observe(on_image_change, names="value")
    
    # Initialize
    subjects = layout.get_subjects()
    subject_dropdown.options = subjects
    if subjects:
        subject_dropdown.value = subjects[0]

    update_sessions(subject_dropdown.value)
    update_images(subject_dropdown.value, session_dropdown.value)
    
    # Display widgets
    display(subject_dropdown, session_dropdown, image_dropdown, output_widget)

# Example usage:
# explore_bids_directory('/path/to/bids/dataset')
