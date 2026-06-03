from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
import uuid
import traceback
import tempfile
from agent.graph import build_graph

app = FastAPI(title="Clinical Agent API")

workflow = build_graph()


@app.post("/generate_summary")
async def generate_summary(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())

    base_temp = tempfile.gettempdir()
    temp_dir = os.path.join(base_temp, "clinical_agent", session_id)
    os.makedirs(temp_dir, exist_ok=True)

    file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        initial_state = {
            "patient_id": session_id,
            "available_pdfs": [file_path],
            "messages": [],
            "read_chunks": [],
            "iteration_count": 0,
            "error_logs": [],
            "tool_executions": [],
            "draft_summary": None,
        }

        final_state = workflow.invoke(initial_state)
        draft = final_state.get("draft_summary")

        # --- THE GRACEFUL FALLBACK ---
        if not draft:
            print(
                "SYSTEM WARNING: Agent failed to generate draft. Returning safety fallback."
            )
            return {
                "status": "warning",
                "total_iterations": final_state.get("iteration_count", 0),
                "agent_chain_of_thought": final_state.get("tool_executions", []),
                "final_summary": {
                    "patient_name": "[EXTRACTION FAILED]",
                    "age": "[EXTRACTION FAILED]",
                    "gender": "[EXTRACTION FAILED]",
                    "admission_date": "[EXTRACTION FAILED]",
                    "discharge_date": "[EXTRACTION FAILED]",
                    "principal_diagnoses": ["[AGENT TIMEOUT - MANUAL REVIEW REQUIRED]"],
                    "secondary_diagnoses": [],
                    "hospital_course": "Agent exceeded iteration limit.",
                    "procedures_performed": [],
                    "discharge_condition": "[UNKNOWN]",
                    "medications_on_admission": [],
                    "discharge_medications": [],
                    "allergies": [],
                    "follow_up_instructions": [],
                    "pending_results": [],
                    "missing_critical_info": ["Entire Document - Agent Timeout"],
                    "identified_conflicts": ["Agent failed to process document"],
                },
            }

        # --- SUCCESS RETURN ---
        return {
            "status": "success",
            "total_iterations": final_state.get("iteration_count", 0),
            "agent_chain_of_thought": final_state.get("tool_executions", []),
            "final_summary": draft.dict(),
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
