from src.imagelib.datasets.dicom import Subjects, Subject, Session, Series

import pandas as pd

dicom_dir: str = "/rsrch5/home/csi/Quarles_Lab/Bajaj_Projects/Melanoma_Data_QIAC/Raw_MRI"
mrn_crosswalk_csv: str = "/rsrch5/home/csi/Quarles_Lab/Bajaj_Projects/Melanoma_Data_QIAC/INFO/2020-1163_ANON-CROSSWALK-02-02-26.csv"
subjects: Subjects = Subjects.from_flat_sessions_dicom_dir(
    dicom_dir,
    mrn_crosswalk_path=mrn_crosswalk_csv,
    series_subdir_pattern="SCANS/",
    dicom_subdir_pattern="DICOM/",
    load_dicom_fields=True
    )

subjects.to_csv("/rsrch5/home/csi/Quarles_Lab/Bajaj_Projects/Melanoma_Data_QIAC/INFO/subjects.csv")