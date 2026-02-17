"""
Prepare a timeline of a patient using DICOM/Nifti data with event data. This timeline can be used to visualize the sequence of events, treatments, and imaging sessions for a patient over time. The timeline can include key dates such as diagnosis, treatment start and end dates, and imaging session dates, allowing for a comprehensive overview of the patient's medical history in relation to their imaging data.
"""
from ..datasets.event import Events
from ..datasets.dicom import Subjects

from pathlib import Path
from typing import Optional, Any, Self
from datetime import date

from pydantic import BaseModel

class SubjectTimelineEntry(BaseModel):
    subject_id: str
    mrn: Optional[str] = None
    entry_date: date
    entry_type: str # e.g. "scan", "treatment", "diagnosis"
    description: Optional[str] = None
    entry: Any
    
    @classmethod
    def from_subjects_and_events(cls, subjects: Subjects, events: Events) -> list[Self]:
        timeline_entries = []
        for subject in subjects.subjects:
            if subject.mrn is None:
                continue
            subject_events = events.get_subject_data(subject_id=subject.mrn)
            for session in subject.sessions:
                if session.study_date is None:
                    continue
                entry_date = session.study_date
                entry_type = "scan"
                description = f"Scan session {session.name}"
                timeline_entries.append(cls(subject_id=subject.mrn, mrn=subject.mrn, entry_date=entry_date, entry_type=entry_type, description=description, entry=session))
            for event_row in subject_events:
                for field in event_row.fields:
                    if field.event_date:
                        timeline_entries.append(cls(subject_id=subject.mrn, mrn=subject.mrn, entry_date=field.event_date, entry_type="event", description=f"{field.name}: {field.value}", entry=field))
        return sorted(timeline_entries, key=lambda x: x.entry_date)