"""
Store clinical data related to subjects and sessions. This may include treatment information, demographics, and other relevant clinical details that can be associated with the imaging data.
"""
from pathlib import Path
from typing import Optional, Any, Self
from datetime import date

from pydantic import BaseModel, Field

class EventDataField(BaseModel):
    name: str
    value: Any
    event_date: Optional[date] = None
    description: Optional[str] = None
    
class EventDataRow(BaseModel):
    """Session-level event data."""
    subject_id: str
    session_id: Optional[str] = None
    fields: list[EventDataField] = Field(default_factory=list)
    
class Events(BaseModel):
    """Container for all event data."""
    rows: list[EventDataRow] = Field(default_factory=list)
    
    @classmethod
    def from_csv(cls, csv_path: Path, date_colname: str = "date") -> Self:
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        df[date_colname] = pd.to_datetime(df[date_colname], errors='coerce').dt.date
        rows = []
        for _, row in df.iterrows():
            subject_id = row['subject_id']
            session_id = row.get('session_id', None)
            fields = [EventDataField(name=col, value=row[col], event_date=row.get(date_colname, None)) for col in df.columns if col not in ['subject_id', 'session_id']]
            rows.append(EventDataRow(subject_id=subject_id, session_id=session_id, fields=fields))
        return cls(rows=rows)
    
    def get_subject_data(self, subject_id: str) -> list[EventDataRow]:
        return [row for row in self.rows if row.subject_id == subject_id]
    
    def filter_by_date_range(self, start_date: date, end_date: date) -> 'Events':
        filtered_rows = []
        for row in self.rows:
            filtered_fields = [field for field in row.fields if field.event_date and start_date <= field.event_date <= end_date]
            if filtered_fields:
                filtered_rows.append(EventDataRow(subject_id=row.subject_id, session_id=row.session_id, fields=filtered_fields))
        return Events(rows=filtered_rows)
    
    def filter_by_fields(self, field_names: list[str]) -> 'Events':
        filtered_rows = []
        for row in self.rows:
            filtered_fields = [field for field in row.fields if field.name in field_names]
            if filtered_fields:
                filtered_rows.append(EventDataRow(subject_id=row.subject_id, session_id=row.session_id, fields=filtered_fields))
        return Events(rows=filtered_rows)