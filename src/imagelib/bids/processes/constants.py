import os

ROOT_DIR: str = os.environ.get("IMAGELIB_HOME", os.getcwd())
if ROOT_DIR is None:
    raise ValueError("IMAGELIB_HOME environment variable not set.")

BIDS_DESC_ENTITY_MNI152: str = "MNI152"
BIDS_DESC_ENTITY_N4BFC: str = "n4bfc"
BIDS_DESC_ENTITY_BRAIN_EXTRACT: str = "brainExtract"
BIDS_DESC_ENTITY_PROB_BRAIN_MASK: str = "probBrainMask"
BIDS_DESC_ENTITY_TISSUE_SEGMENT: str = "tissueSegment"
# BIDS_DESC_ENTITY_PARCELLATION: str = "parcellation"
BIDS_DESC_ENTITY_PARCELLATION_HARVARD_OXFORD: str = "harvardOxford"
BIDS_DESC_ENTITY_PARCELLATION_DKT: str = "desikanKillianyTourville"
# BIDS_DESC_ENTITY_CORTICAL_THICKNESS: str = "corticalThickness"
BIDS_DESC_ENTITY_CORTICAL_THICKNESS: str = "kellyKapowski"
BIDS_DESC_ENTITY_CORTICAL_THICKNESS_REGION_STATS: str = "corticalThicknessRegionStats"
BIDS_DESC_ENTITY_SEGMENT_REGION_STATS: str = "segmentRegionStats"