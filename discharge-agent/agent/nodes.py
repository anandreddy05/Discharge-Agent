from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from schema.output_models import DischargeSummaryDraft
from schema.agent_state import AgentState
from agent.tools import AGENT_TOOLS
import os
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
import re

load_dotenv(override=True)

base_url = os.getenv("base_url")


def create_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
    if not api_key:
        raise RuntimeError(
            "OpenAI credentials are missing. Set OPENAI_API_KEY or OPENAI_ADMIN_KEY before using the model."
        )
    return ChatOpenAI(model="gpt-4o", temperature=0, base_url=base_url, api_key=api_key)


def get_llm_with_tools():
    llm = create_llm()
    return llm.bind_tools(AGENT_TOOLS + [DischargeSummaryDraft])



def validator(state: AgentState):
    """
    STRICT RUBRIC VALIDATOR: Enforces no fabrication, med rec checks, and programmatic conflict detection.
    """
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

    # =========================================================
    # 1. RECONSTRUCT GROUND TRUTH
    # =========================================================
    all_read_text = ""
    for msg in state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "tool":
            if "CONTENT FOR PAGES" in str(msg.content):
                all_read_text += str(msg.content) + "\n"

    all_read_text_lower = all_read_text.lower()
    critique_issues = []

    # =========================================================
    # 2. ANTI-HALLUCINATION CHECKS (Name & Diagnoses)
    # =========================================================
    if draft.patient_name and "[MISSING" not in draft.patient_name:
        if len(draft.patient_name) <= 2 and draft.patient_name.isalpha():
            critique_issues.append(
                f"SUSPECT: Patient name '{draft.patient_name}' looks like a gender marker or typo. Verify or mark [MISSING - FLAG FOR REVIEW]."
            )
        else:
            name_parts = [p.lower() for p in draft.patient_name.split() if len(p) > 2]
            if name_parts and not any(
                part in all_read_text_lower for part in name_parts
            ):
                critique_issues.append(
                    f"HALLUCINATION: Patient name '{draft.patient_name}' not found in read text. Use [MISSING - FLAG FOR REVIEW]."
                )

    # Check Diagnoses
    for dx in draft.principal_diagnoses:
        if "[MISSING" not in dx:
            words = [w.lower() for w in dx.replace("-", " ").split() if len(w) > 4]
            if words and not any(w in all_read_text_lower for w in words):
                critique_issues.append(
                    f"HALLUCINATION: Diagnosis '{dx}' seems fabricated. Key terms not found in text."
                )

    # =========================================================
    # 3. RUBRIC ENFORCEMENT: MED REC & LAZY ARRAYS
    # =========================================================
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
                    f"RUBRIC FAIL: Medication {med.name} was {med.action} with no reason, but you didn't flag it in missing_critical_info."
                )

    if len(draft.medications_on_admission) == 0:
        critique_issues.append(
            "RUBRIC FAIL: Admission meds array is empty. If missing, output '[MISSING - FLAG FOR REVIEW]'."
        )

    # =========================================================
    # 4. CONFLICT DETECTION (Programmatic Regex Check)
    # =========================================================
    lab_patterns = {
        "haemoglobin": r"haemoglobin[\s\(A-Za-z\)]*?[:\-]?\s*(\d+\.?\d*)",
        "wbc": r"wbc.*?(\d{1,3},?\d{3})",
        "platelet": r"platelet.*?(\d{1,3},?\d{3})",
    }

    for lab_name, pattern in lab_patterns.items():
        # Find all matches in the document
        matches = re.findall(pattern, all_read_text_lower)
        unique_matches = list(set(matches))

        # If multiple different values exist for the same lab, it's a conflict
        if len(unique_matches) > 1:
            conflict_msg = f"Conflicting {lab_name} values found: {unique_matches}"

            # Check if the agent correctly identified this conflict in its draft
            if not any(lab_name in flag.lower() for flag in draft.identified_conflicts):
                critique_issues.append(
                    f"CONFLICT NOT FLAGGED: {conflict_msg}. You MUST add this to identified_conflicts."
                )

    # =========================================================
    # 5. PASS/FAIL ROUTING
    # =========================================================
    missing_fields = [
        field for field in draft.missing_critical_info if "[MISSING" not in field
    ]

    if not critique_issues and not missing_fields:
        success_msg = ToolMessage(
            content="Validation Passed! Summary finalized and strictly verified.",
            tool_call_id=tool_call_id,
        )
        return {"messages": [success_msg], "validator_critique": "PASS"}

    # DYNAMIC CRITIQUE
    read_chunks = state.get("read_chunks", [])
    critique = "VALIDATION FAILED against Rubric Rules:\n"
    for issue in critique_issues:
        critique += f"- {issue}\n"
    if missing_fields:
        critique += f"- Still missing: {', '.join(missing_fields)}\n"

    critique += f"\nYou have already read: {read_chunks}. "
    critique += "Read a new 5-page chunk to find this data, or explicitly flag it as [MISSING - FLAG FOR REVIEW]."

    # Answer all parallel tools
    tool_messages = []
    for tc in last_message.tool_calls:
        if tc["name"] == "DischargeSummaryDraft":
            tool_messages.append(ToolMessage(content=critique, tool_call_id=tc["id"]))
        else:
            tool_messages.append(
                ToolMessage(content="Action intercepted.", tool_call_id=tc["id"])
            )

    return {"messages": tool_messages, "validator_critique": critique}


def reasoner(state: AgentState):
    """
    The main intelligence node. Uses semantic chunking and tracks read history.
    """
    # 1. Fetch memory of what has already been read
    already_read = state.get("read_chunks", [])
    read_history = ", ".join(already_read) if already_read else "None yet."

    # Safely extract the file path
    file_path = state.get("available_pdfs", [""])[0]

    # 2. Build the System Message
    sys_msg = SystemMessage(
        content=f"""
You are an elite Clinical AI drafting a flawless Discharge Summary.

FILE TO PROCESS: {file_path}
PAGES ALREADY READ: {read_history}

STEP 1: Call `get_document_info` to find the TOTAL_PAGES.

STEP 2: Use `read_document_pages` to read the file. (STRICT LIMIT: 5 pages per call).
- PASS 1: Read pages 1 to 5 (Front matter, Diagnoses, Discharge Meds).
- PASS 2: Read the final 5 pages (Back matter, Admission Meds).
- PASS 3: Calculate and read the middle 5 pages (Labs, Hospital Course).

STEP 3: CLINICAL SAFETY & ESCALATION (CRITICAL)
- If you extract a list of Discharge Medications, you MUST call `check_drug_interaction` to verify they are safe together.
- If you find a severe conflict (like contradictory lab values or a dangerous drug interaction), you MUST call `flag_for_clinician_review` to officially log the escalation.

STEP 4: THE ZERO-HALLUCINATION EXTRACTION CHECKLIST
Before drafting, verify you have hunted for these notoriously hidden items:
- ADMISSION MEDICATIONS: Look for phrases like "K/c/o", "Past Medical History", or "Ayurvedic".
- DISCHARGE MEDICATIONS: Look for tables labeled "ADVICE ON DISCHARGE".
- DATES: Look at the headers and footers of the first and last pages.

STEP 5: If you have checked the front, back, and middle, and a piece of data is TRULY missing from the text, DO NOT GUESS. Safely output "[MISSING - FLAG FOR REVIEW]" in your draft. 

Call `DischargeSummaryDraft` immediately when you have satisfied all safety checks and gathered your data.
"""
    )

    messages = [sys_msg] + state.get("messages", [])
    llm_with_tools = get_llm_with_tools()
    response = llm_with_tools.invoke(messages)

    # --- OBSERVABILITY & TRACING ---
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
                    "result": None 
                }
            )

            if tc["name"] == "DischargeSummaryDraft":
                # ONLY update the draft if this specific tool is called
                draft = DischargeSummaryDraft(**tc["args"])
                
            elif tc["name"] == "read_document_pages":
                chunk_log = f"Pages {tc['args']['start_page']}-{tc['args']['end_page']}"
                new_chunks.append(chunk_log)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "current_plan": current_plan,
        "current_step": "reasoning",
        "read_chunks": new_chunks,
        "tool_executions": executions,
        "draft_summary": draft, 
    }