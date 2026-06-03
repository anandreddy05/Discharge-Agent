from typing import TypedDict, Optional, List, Dict, Any, Annotated
import operator
from langgraph.graph.message import add_messages
from schema.output_models import DischargeSummaryDraft

class AgentState(TypedDict):
    # --- System Inputs ---
    patient_id: str
    available_pdfs: List[str]
    
    # --- Conversation & Tool Memory ---
    messages: Annotated[list, add_messages]
    read_chunks: Annotated[List[str], operator.add]
    extracted_labs: Annotated[List[Dict[str, str]], operator.add] 
    validator_critique: str 
    
    # --- Execution Tracking ---
    iteration_count: int
    current_plan: str
    current_step: str
    error_logs: List[str]
    tool_executions: List[Dict[str, Any]]
        
    # --- Final Output ---
    draft_summary: Optional[DischargeSummaryDraft]