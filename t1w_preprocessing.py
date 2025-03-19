from src.imagelib.bids.processes.preprocess import n4_bias_field_correction, brain_extraction_antspynet
from src.imagelib.bids.pipeline import BIDSPipeline, BIDSPipelineTree, get_new_pipeline_derived_filename, Process

from pathlib import Path, PosixPath

from bids import BIDSLayout

bids_root: PosixPath = Path("/Users/cmokashi/data/bids_datasets/open_neuro/ds005596-1.1.1/")
bids_layout = BIDSLayout(bids_root, derivatives=True)

processes: list[Process] = [
    Process(name="n4_bias_field_correction", description="N4 Bias Field Correction", logic=n4_bias_field_correction),
    Process(name="brain_extraction_antspynet", description="Brain Extraction using ANTsPyNet", logic=brain_extraction_antspynet, kwargs={"modality": "t1"})
]
tree: BIDSPipelineTree = BIDSPipelineTree()
pipeline_name: str = "t1w_preprocessed"
tree.set_default_values(pipeline_name)
pipeline: BIDSPipeline = BIDSPipeline(
    name=pipeline_name,
    bids_root=bids_root,
    tree=tree,
    description="T1w Preprocessing Pipeline",
    bids_filters={"suffix": "T1w", "extension": ".nii.gz"},
    processes=processes,
    is_chain=True,
    overwrite_files=False
)
pipeline.create_tree()

pipeline.execute()