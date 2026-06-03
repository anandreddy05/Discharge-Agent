from pydantic import BaseModel, Field
from typing import List, Literal

class ReconciledMedication(BaseModel):
    name: str = Field(description="Name of the medication.")
    dosage: str = Field(default="[NOT SPECIFIED]", description="Dosage amount. Use [NOT SPECIFIED] if absent.")
    frequency: str = Field(default="[NOT SPECIFIED]", description="Frequency of administration.")
    action: Literal["Continued", "Started", "Stopped", "Changed"] = Field(
        description="Status compared to admission. If it is only on the discharge list, it is 'Started'. If it was on admission but not discharge, it is 'Stopped'."
    )
    reason_for_change: str = Field(
        description="Why the medication was changed, started, or stopped. If no reason is explicitly documented in the notes, you MUST output 'No documented reason'."
    )

class PendingResult(BaseModel):
    test_name: str
    status: str = Field(default="Pending", description="Current status of the lab or procedure.")

class DischargeSummaryDraft(BaseModel):
    # Demographics
    patient_name: str = Field(default="[MISSING - FLAG FOR REVIEW]")
    age: str = Field(default="[MISSING - FLAG FOR REVIEW]")
    gender: str = Field(default="[MISSING - FLAG FOR REVIEW]")
    admission_date: str = Field(default="[MISSING - FLAG FOR REVIEW]")
    discharge_date: str = Field(default="[MISSING - FLAG FOR REVIEW]")
    
    # Diagnoses
    principal_diagnoses: List[str] = Field(description="Primary reasons for admission.")
    secondary_diagnoses: List[str] = Field(description="Other identified conditions or comorbidities.")
    
    # Clinical Narrative
    hospital_course: str = Field(
        description="Chronological summary of the hospital stay. Do not invent details. Base strictly on provided notes."
    )
    procedures_performed: List[str] = Field(default_factory=list)
    discharge_condition: str = Field(default="[NOT EXPLICITLY DOCUMENTED]")
    
    # Medications & Instructions
    medications_on_admission: List[str] = Field(
        default_factory=list,
        description="Medications the patient was taking prior to admission. Use ['[NOT DOCUMENTED]'] if none are explicitly stated."
    )
    discharge_medications: List[ReconciledMedication]
    allergies: List[str] = Field(
        default_factory=lambda: ["[NOT DOCUMENTED]"], 
        description="List documented allergies. If none found, output '[NOT DOCUMENTED]'."
    )
    follow_up_instructions: List[str]
    
    # The Core Safety Guardrails
    pending_results: List[PendingResult] = Field(
        default_factory=list, 
        description="List any labs, cultures, or scans marked as awaited or pending."
    )
    missing_critical_info: List[str] = Field(
        default_factory=list, 
        description="Explicitly list any required section (like Age or Gender) that could not be found."
    )
    identified_conflicts: List[str] = Field(
        default_factory=list, 
        description="Explicitly list conflicting information found between different source notes (e.g., differing diagnosis reports)."
    )