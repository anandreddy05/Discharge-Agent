from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from schema.output_models import DischargeSummaryDraft
from schema.agent_state import AgentState
from agent.tools import AGENT_TOOLS
from learning.memory import format_memory_for_prompt
import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

base_url = os.getenv("base_url")


def create_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
    if not api_key:
        raise RuntimeError("OpenAI credentials missing.")
    return ChatOpenAI(model="gpt-4o", temperature=0, base_url=base_url, api_key=api_key)


def get_llm_with_tools():
    return create_llm().bind_tools(AGENT_TOOLS + [DischargeSummaryDraft])


def validator(state: AgentState):
    last_message = state["messages"][-1]

    draft_tool_call = next(
        (tc for tc in last_message.tool_calls if tc["name"] == "DischargeSummaryDraft"),
        None,
    )

    if not draft_tool_call:
        return {"validator_critique": "Error: No draft tool call found."}

    tool_call_id = draft_tool_call["id"]
    draft = state.get("draft_summary")

    if not draft:
        return {
            "messages": [
                ToolMessage(content="Draft failed.", tool_call_id=tool_call_id)
            ],
            "validator_critique": "Draft failed. Try again.",
        }

    all_read_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "tool":
            if "CONTENT FOR PAGES" in str(msg.content):
                all_read_text += str(msg.content) + "\n"

    all_read_text_lower = all_read_text.lower()
    critique_issues = []

    if draft.patient_name and "[MISSING" not in draft.patient_name:
        if len(draft.patient_name) <= 2 and draft.patient_name.isalpha():
            critique_issues.append(
                f"SUSPECT: Patient name '{draft.patient_name}' looks like a gender marker."
            )
        else:
            name_parts = [p.lower() for p in draft.patient_name.split() if len(p) > 2]
            if name_parts and not any(
                part in all_read_text_lower for part in name_parts
            ):
                critique_issues.append(
                    f"HALLUCINATION: Patient name '{draft.patient_name}' not found in read text."
                )

    for dx in draft.principal_diagnoses:
        if "[MISSING" not in dx:
            words = [w.lower() for w in dx.replace("-", " ").split() if len(w) > 4]
            if words and not any(w in all_read_text_lower for w in words):
                critique_issues.append(
                    f"HALLUCINATION: Diagnosis '{dx}' not found in text."
                )

    for med in draft.discharge_medications:
        if (
            med.action in ["Started", "Stopped", "Changed"]
            and "no documented reason" in med.reason_for_change.lower()
        ):
            flagged = any(
                med.name.lower() in flag.lower() for flag in draft.missing_critical_info
            )
            if not flagged:
                critique_issues.append(
                    f"RUBRIC FAIL: {med.name} was {med.action} with no reason but not flagged."
                )

    if len(draft.medications_on_admission) == 0:
        critique_issues.append("RUBRIC FAIL: Admission meds array is empty.")

    lab_patterns = {
        "haemoglobin": r"haemoglobin[\s\(A-Za-z\)]*?[:\-]?\s*(\d+\.?\d*)",
        "wbc": r"wbc.*?(\d{1,3},?\d{3})",
        "platelet": r"platelet.*?(\d{1,3},?\d{3})",
    }

    for lab_name, pattern in lab_patterns.items():
        matches = list(set(re.findall(pattern, all_read_text_lower)))
        if len(matches) > 1:
            if not any(lab_name in flag.lower() for flag in draft.identified_conflicts):
                critique_issues.append(
                    f"CONFLICT NOT FLAGGED: Conflicting {lab_name} values {matches}."
                )

    missing_fields = [f for f in draft.missing_critical_info if "[MISSING" not in f]

    if not critique_issues and not missing_fields:
        return {
            "messages": [
                ToolMessage(
                    content="Validation Passed! Summary finalized.",
                    tool_call_id=tool_call_id,
                )
            ],
            "validator_critique": "PASS",
        }

    read_chunks = state.get("read_chunks", [])
    critique = "VALIDATION FAILED:\n" + "".join(f"- {i}\n" for i in critique_issues)
    if missing_fields:
        critique += f"- Still missing: {', '.join(missing_fields)}\n"
    critique += f"\nAlready read: {read_chunks}. Read a new chunk or flag as [MISSING - FLAG FOR REVIEW]."

    tool_messages = []
    for tc in last_message.tool_calls:
        content = (
            critique if tc["name"] == "DischargeSummaryDraft" else "Action intercepted."
        )
        tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    return {"messages": tool_messages, "validator_critique": critique}


def reasoner(state: AgentState):
    already_read = state.get("read_chunks", [])
    read_history = ", ".join(already_read) if already_read else "None yet."
    file_path = state.get("available_pdfs", [""])[0]

    existing_draft = state.get("draft_summary")

    memory_block = ""
    if existing_draft:
        try:
            memory_block = format_memory_for_prompt(existing_draft.dict())
        except Exception:
            memory_block = ""

    sys_msg = SystemMessage(
        content=f"""
You are an elite Clinical AI drafting a flawless Discharge Summary.
 
FILE TO PROCESS: {file_path}
PAGES ALREADY READ: {read_history}
{memory_block}

CRITICAL FORMATTING RULES (MUST FOLLOW):
- 'allergies' must ALWAYS be a JSON array (list), even for single or no values.
  ✅ CORRECT: "allergies": ["[NOT DOCUMENTED]"]
  ❌ WRONG: "allergies": "[NOT DOCUMENTED]"
- 'medications_on_admission' must ALWAYS be a JSON array (list).
  ✅ CORRECT: "medications_on_admission": ["[NOT DOCUMENTED]"]
  ❌ WRONG: "medications_on_admission": "[NOT DOCUMENTED]"
- 'pending_results' must ALWAYS be a JSON array of objects.
- 'discharge_medications' must ALWAYS be a JSON array of objects.

STEP 1: Call `get_document_info` to find TOTAL_PAGES.
 
STEP 2: Use `read_document_pages` (STRICT LIMIT: 5 pages per call).
- PASS 1: Pages 1–5 (Front matter, Diagnoses, Discharge Meds).
- PASS 2: Final 5 pages (Admission Meds).
- PASS 3: Middle 5 pages (Labs, Hospital Course).
 
STEP 3: CLINICAL SAFETY
- After extracting discharge meds, call `check_drug_interaction`.
- For severe conflicts or dangerous interactions, call `flag_for_clinician_review`.
 
STEP 4: ZERO-HALLUCINATION CHECKLIST
- ADMISSION MEDS: Look for "K/c/o", "Past Medical History", "Ayurvedic".
- DISCHARGE MEDS: Look for "ADVICE ON DISCHARGE" tables.
- DATES: Check headers and footers of first and last pages.
 
STEP 5: If data is truly absent, output "[MISSING - FLAG FOR REVIEW]". Never guess.
 
Call `DischargeSummaryDraft` once all safety checks are satisfied.
"""
    )

    messages = [sys_msg] + state.get("messages", [])
    response = get_llm_with_tools().invoke(messages)

    current_plan = (
        response.content.strip() if response.content else "Executing tool calls."
    )
    draft = state.get("draft_summary")
    new_chunks = []
    executions = state.get("tool_executions", [])

    if len(state.get("messages", [])) >= 2:
        last_msg = state["messages"][-1]
        if last_msg.type == "tool":
            for exec_log in executions:
                if exec_log.get("result") is None:
                    exec_log["result"] = str(last_msg.content)[:200] + "... [TRUNCATED]"

    if response.tool_calls:
        for tc in response.tool_calls:
            executions.append(
                {
                    "iteration": state.get("iteration_count", 0) + 1,
                    "reasoning": current_plan,
                    "action": tc["name"],
                    "args": tc["args"],
                    "result": None,
                }
            )
            if tc["name"] == "DischargeSummaryDraft":
                draft = DischargeSummaryDraft(**tc["args"])
            elif tc["name"] == "read_document_pages":
                new_chunks.append(
                    f"Pages {tc['args']['start_page']}-{tc['args']['end_page']}"
                )

        return {
            "messages": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_plan": current_plan,
            "current_step": "reasoning",
            "read_chunks": new_chunks,
            "tool_executions": executions,
            "draft_summary": draft,
        }
