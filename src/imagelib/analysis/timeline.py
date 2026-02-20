"""
Prepare a timeline of a patient using DICOM/Nifti data with event data. This timeline can be used to visualize the sequence of events, treatments, and imaging sessions for a patient over time. The timeline can include key dates such as diagnosis, treatment start and end dates, and imaging session dates, allowing for a comprehensive overview of the patient's medical history in relation to their imaging data.
"""
from ..datasets.event import Events, EventDataRow
from ..datasets.dicom import MRSubjects, MRSubject, Session

from pathlib import Path
from typing import Optional, Any, Self
from datetime import date

from pydantic import BaseModel, Field

class PatientTimelineEntry(BaseModel):
    subject_id: str
    mrn: Optional[str] = None
    entry_date: date
    entry_type: str # e.g. "scan", "treatment", "diagnosis"
    description: Optional[str] = None
    entry: Any
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PatientTimelineEntry):
            return NotImplemented
        return (self.subject_id == other.subject_id and
                self.entry_date == other.entry_date and
                self.entry_type == other.entry_type and
                self.entry == other.entry)
    
class PatientTimeline(BaseModel):
    entries: list[PatientTimelineEntry] = Field(default_factory=list)
    
    def __getitem__(self, key: str) -> list[PatientTimelineEntry]:
        if key == "entries":
            return self.entries
        raise KeyError(f"Key {key} not found in PatientTimeline")
    
    def __str__(self) -> str:
        output = []
        for entry in self.entries:
            output.append(f"Subject {entry.subject_id}, Date: {entry.entry_date}, Type: {entry.entry_type}, Description: {entry.description}")
        return "\n".join(output)
    
    def __add__(self, other: Self) -> 'PatientTimeline':
        if not isinstance(other, PatientTimeline):
            return NotImplemented
        combined_entries = self.entries + other.entries
        combined_entries.sort(key=lambda x: x.entry_date)
        return PatientTimeline(entries=combined_entries)
    
    def __sub__(self, other: Self) -> 'PatientTimeline':
        if not isinstance(other, PatientTimeline):
            return NotImplemented
        remaining_entries = [entry for entry in self.entries if entry not in other.entries]
        return PatientTimeline(entries=remaining_entries)
    
    @property
    def types(self) -> set[str]:
        return set(entry.entry_type for entry in self.entries)
    
    def add_scans(self, subjects: MRSubjects):
        """Add scan sessions from DICOM data to the timeline."""
        for subject in subjects.iterate():
            for session in subject.sessions:
                if not session.study_date:
                    continue
                self.entries.append(PatientTimelineEntry(
                    subject_id=subject.name,
                    mrn=subject.mrn,
                    entry_date=session.study_date,
                    entry_type="scan",
                    description=f"Scan session {session.name}",
                    entry=session
                ))
    
    def add_events(self, events: Events, subjects_reference: Optional[MRSubjects] = None):
        """Add clinical events from event data to the timeline."""
        for event_row in events.iterate():
            if not event_row.event_date:
                continue
            if not event_row.subject_id and subjects_reference:
                # Try to find subject_id using MRN if subject_id is not provided
                mrn = event_row.mrn
                if mrn and isinstance(mrn, str):
                    subject: Optional[MRSubject] = subjects_reference.get_subject_by_mrn(mrn)
                    if subject:
                        event_row.subject_id = subject.name
            if not event_row.subject_id:
                continue
            self.entries.append(PatientTimelineEntry(
                subject_id=event_row.subject_id,
                mrn=event_row.mrn,
                entry_date=event_row.event_date,
                entry_type="event",
                description=f"Clinical event for session {event_row.session_id}",
                entry=event_row
            ))
    
    def iterate(self):
        for entry in self.entries:
            yield entry
    
    def get_timeline_for_subject(self, subject_id: str) -> list[PatientTimelineEntry]:
        return [entry for entry in self.entries if entry.subject_id == subject_id]
    
    def filter_timeline_by_date_range(self, start_date: date, end_date: date) -> 'PatientTimeline':
        filtered_entries = [entry for entry in self.entries if start_date <= entry.entry_date <= end_date]
        return PatientTimeline(entries=filtered_entries)
    
    def filter_timeline_by_entry_type(self, entry_types: list[str]) -> 'PatientTimeline':
        filtered_entries = [entry for entry in self.entries if entry.entry_type in entry_types]
        return PatientTimeline(entries=filtered_entries)
    
    def remove_entries_by_event_fields(self, fields: dict[str, Any]) -> 'PatientTimeline':
        """
        Remove timeline entries that match specific event field values. 
        For example, if you want to remove all events where "treatment_type" is "chemotherapy", 
            you can call this method with fields={"treatment_type": "chemotherapy"}. 
        This will return a new PatientTimeline with those entries removed.
        
        Args:
            fields (dict[str, Any]): A dictionary where keys are event field names and values are the values to match for removal. 
                If an event entry has a field that matches any of the specified field-value pairs, it will be removed from the timeline.
        """
        filtered_entries = []
        for entry in self.entries:
            if entry.entry_type != "event":
                filtered_entries.append(entry)
                continue
            if not isinstance(entry.entry, EventDataRow):
                filtered_entries.append(entry)
                continue
            match = False
            for field_name, field_value in fields.items():
                if entry.entry.get_field_value(field_name) == field_value:
                    match = True
                    break
            if not match:
                filtered_entries.append(entry)
        return PatientTimeline(entries=filtered_entries)
    
    def filter_timeline_by_event_fields(self, fields: dict[str, Any]) -> 'PatientTimeline':
        filtered_entries = []
        for entry in self.entries:
            if entry.entry_type != "event":
                continue
            if not isinstance(entry.entry, EventDataRow):
                continue
            match = True
            for field_name, field_value in fields.items():
                if entry.entry.get_field_value(field_name) != field_value:
                    match = False
                    break
            if match:
                filtered_entries.append(entry)
        return PatientTimeline(entries=filtered_entries)
    
    def get_timeline_summary_for_subject(self, subject_id: str) -> str:
        patient_entries = self.get_timeline_for_subject(subject_id)
        patient_entries.sort(key=lambda x: x.entry_date)
        start_date = patient_entries[0].entry_date if patient_entries else None
        output = [f"Timeline for Subject {subject_id}:"]
        for entry in patient_entries:
            current_date = entry.entry_date
            days_since_start = (current_date - start_date).days if start_date else "N/A"
            if entry.entry_type == "scan":
                output_str: str = f"  - {entry.entry_date} (Day {days_since_start}): Scan session"
                if isinstance(entry.entry, Session):
                    for series in entry.entry.iterate():
                        output_str += f"\n    - Series {series.name} with {series.n_scans} instances"
                output.append(output_str)
            elif entry.entry_type == "event":
                output_str: str = f"  - {entry.entry_date} (Day {days_since_start}): {entry.entry.event_name if isinstance(entry.entry, EventDataRow) else 'Clinical event'}"
                if isinstance(entry.entry, EventDataRow):
                    for field in entry.entry.fields:
                        output_str += f"\n    - {field.name}: {field.value} (Description: {field.description})"
                output.append(output_str)
        return "\n\n".join(output)