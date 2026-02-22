"""
Store clinical data related to subjects and sessions. This may include treatment information, demographics, and other relevant clinical details that can be associated with the imaging data.
"""
from pathlib import Path
from typing import Optional, Any, Self
from datetime import date

from pydantic import BaseModel, Field, ValidationError, model_validator
import pandas as pd

class EventDataField(BaseModel):
    name: str
    value: Any
    description: Optional[str] = None
    
class EventDataRow(BaseModel):
    """Session-level event data."""
    subject_id: Optional[str] = None
    mrn: Optional[str] = None
    session_id: Optional[str] = None
    event_date: Optional[date] = None
    event_name: Optional[str] = None
    fields: list[EventDataField] = Field(default_factory=list)
    
    def get_field_value(self, field_name: str) -> Optional[Any]:
        for field in self.fields:
            if field.name == field_name:
                return field.value
        return None
    
class Events(BaseModel):
    """Container for all event data."""
    rows: list[EventDataRow] = Field(default_factory=list)
    
    def __str__(self) -> str:
        output = []
        for row in self.rows:
            output.append(f"Subject {row.subject_id}, Session {row.session_id}, Event Date: {row.event_date}, Event Name: {row.event_name}")
            for field in row.fields:
                output.append(f"  - {field.name}: {field.value}, Description: {field.description})")
        return "\n".join(output)
    
    @classmethod
    def from_csv(cls, csv_path: str | Path, date_colname: str = "date", subject_colname: Optional[str] = None, mrn_colname: Optional[str] = None, session_colname: Optional[str] = None, event_name_colname: Optional[str] = None, read_file_kwargs: Optional[dict] = None) -> Self:
        if read_file_kwargs is None:
            read_file_kwargs = {}
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path, **read_file_kwargs)
        # Convert NaN to None
        df = df.where(pd.notnull(df), None)
        df[date_colname] = pd.to_datetime(df[date_colname], errors='coerce').dt.date
        if all(colname is None for colname in [subject_colname, mrn_colname]):
            raise ValueError("At least one of subject_colname or mrn_colname must be provided to identify subjects.")
        rows = []
        for row_idx, row in df.iterrows():
            subject_id = row.get(subject_colname, None) if subject_colname else None
            session_id = row.get(session_colname, None) if session_colname else None
            mrn = row.get(mrn_colname, None) if mrn_colname else None
            event_date = row[date_colname] if pd.notnull(row[date_colname]) else None
            event_name = row.get(event_name_colname, None) if event_name_colname else None
            fields = []
            for prop in [subject_id, session_id, mrn, event_name]:
                if prop is not None:
                    if not isinstance(prop, str):
                        prop = str(prop)
                    if pd.isna(prop):
                        prop = None
            for colname in df.columns:
                if colname not in [date_colname, subject_colname, mrn_colname, session_colname, event_name_colname]:
                    fields.append(EventDataField(name=colname, value=row[colname], description=None))
            try:
                rows.append(EventDataRow(subject_id=subject_id, mrn=mrn, session_id=session_id, event_date=event_date, event_name=event_name, fields=fields))
            except ValidationError as e:
                print(f"Validation error for row {row_idx} with subject_id={subject_id}, session_id={session_id}, event_date={event_date}: {e}")
                for field in [subject_colname, session_colname, date_colname, mrn_colname, event_name_colname]:
                    if field:
                        print(f"  - {field}: {row.get(field, None)} - type {type(row.get(field, None))}")
                raise e
        return cls(rows=rows)

    def iterate(self):
        for row in self.rows:
            yield row
    
    def get_subject_data(self, subject_id: str) -> list[EventDataRow]:
        return [row for row in self.rows if row.subject_id == subject_id]
    
    def filter_by_date_range(self, start_date: date, end_date: date) -> 'Events':
        filtered_rows = []
        for row in self.rows:
            filtered_fields = [field for field in row.fields if row.event_date and start_date <= row.event_date <= end_date]
            if filtered_fields:
                filtered_rows.append(EventDataRow(subject_id=row.subject_id, session_id=row.session_id, event_date=row.event_date, event_name=row.event_name, fields=filtered_fields))
        return Events(rows=filtered_rows)
    
    def filter_by_fields(self, field_names: list[str]) -> 'Events':
        filtered_rows = []
        for row in self.rows:
            filtered_fields = [field for field in row.fields if field.name in field_names]
            if filtered_fields:
                filtered_rows.append(EventDataRow(subject_id=row.subject_id, session_id=row.session_id, event_date=row.event_date, event_name=row.event_name, fields=filtered_fields))
        return Events(rows=filtered_rows)