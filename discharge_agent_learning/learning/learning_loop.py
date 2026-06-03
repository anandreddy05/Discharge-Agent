import argparse
import json
import os
import uuid
import shutil
import tempfile
from datetime import datetime, timezone

from agent.graph import build_graph
from learning.reviewer import simulate_doc_review
from learning.reward import compute_reward, reward_summary, SCORED_SECTIONS
from learning.memory import store_corrections, memory_stats
 
RESULTS_LOG = os.path.join(os.getcwd(), "learning_results.json")

def _collect_source_text(final_state: dict) -> str:
    """Concatenate all read_document_pages tool outputs from message history."""
    parts = []
    for msg in final_state.get("messages", []):
        if hasattr(msg, "type") and msg.type == "tool":
            content = str(msg.content)
            if "CONTENT FOR PAGES" in content:
                parts.append(content)
    return "\n".join(parts)
 
def _load_results() -> list:
    if not os.path.exists(RESULTS_LOG):
        return []
    with open(RESULTS_LOG, "r", encoding="utf-8") as f:
        return json.load(f)

 
def _save_results(results: list) -> None:
    with open(RESULTS_LOG, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)



def run_one_iteration(pdf_path: str, iteration_num: int,previous_draft: dict = None) -> dict:
    """
    Runs the full agent → reviewer → reward cycle once.
    Returns a result dict with scores and drafts.
    """
    session_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration_num}  |  session={session_id[:8]}")
    print(f"{'='*60}")

    base_temp = tempfile.gettempdir()
    temp_dir = os.path.join(base_temp, "learning_loop", session_id)
    os.makedirs(temp_dir, exist_ok=True)
    temp_pdf = os.path.join(temp_dir, os.path.basename(pdf_path))
    shutil.copy2(pdf_path, temp_pdf)

    try:
        # 2. Run agent
        workflow = build_graph()
        initial_state = {
            "patient_id": session_id,
            "available_pdfs": [temp_pdf],
            "messages": [],
            "read_chunks": [],
            "iteration_count": 0,
            "error_logs": [],
            "tool_executions": [],
            "draft_summary": previous_draft,
        }
        print("  [1/4] Running agent...")
        final_state = workflow.invoke(initial_state)

        draft_obj = final_state.get("draft_summary")
        if not draft_obj:
            print("  [!] Agent failed to produce a draft. Skipping iteration.")
            return {"iteration": iteration_num, "status": "agent_failed"}
 
        agent_draft = draft_obj.dict()
        print(f"  [1/4] Draft produced. Patient: {agent_draft.get('patient_name')}")

        source_text = _collect_source_text(final_state)
        
        print("  [2/4] Running simulated doctor review...")
        doctor_draft = simulate_doc_review(agent_draft, source_text=source_text)

        print("  [3/4] Computing reward scores...")
        scores = compute_reward(agent_draft, doctor_draft)
        print(reward_summary(scores))

        print("  [4/4] Storing corrections to memory bank...")
        n_stored = store_corrections(agent_draft, doctor_draft, scores, patient_id=session_id)
        print(f"  Stored {n_stored} new correction(s).")

        result = {
            "iteration": iteration_num,
            "session_id": session_id,
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scores": scores,
            "agent_draft": agent_draft,
            "doctor_draft": doctor_draft,
            "agent_iterations": final_state.get("iteration_count", 0),
            "memory_stats_after": memory_stats(),
        }
        return result
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
 


def run_learning_loop(pdf_path: str, n_iterations: int, out_path: str) -> None:
    results = _load_results()
    start_iter = len(results) + 1
    previous_draft = None
    for i in range(start_iter, start_iter + n_iterations):
        result = run_one_iteration(pdf_path, iteration_num=i, previous_draft=previous_draft)
        results.append(result)
        _save_results(results)

        if result.get("status") == "success":
            previous_draft = result.get("agent_draft")
            print("  Memory will carry over to next iteration")
        else:
            print(f"  Iteration {i} failed, keeping previous memory")
        
        print(f"\n  Results saved to {out_path}")
    # Final summary
    successful = [r for r in results if r.get("status") == "success"]
    if len(successful) >= 2:
        first_overall = successful[0]["scores"]["overall"]
        last_overall = successful[-1]["scores"]["overall"]
        delta = last_overall - first_overall
        print(f"\n{'='*60}")
        print(f"LEARNING SUMMARY over {len(successful)} iterations:")
        print(f"  First overall score : {first_overall:.4f}")
        print(f"  Latest overall score: {last_overall:.4f}")
        print(f"  Delta               : {delta:+.4f}")
        print(f"{'='*60}")
    else:
        print("\nNot enough successful iterations to compare yet.")
 
    # Write final results to requested out path
    if out_path != RESULTS_LOG:
        shutil.copy2(RESULTS_LOG, out_path)
        print(f"Results also saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Part 2 learning loop")
    parser.add_argument("--pdf", required=True, help="Path to patient PDF")
    parser.add_argument("--iterations", type=int, default=3, help="Number of iterations to run")
    parser.add_argument("--out", default=RESULTS_LOG, help="Output JSON path for results")
    args = parser.parse_args()
 
    if not os.path.exists(args.pdf):
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
 
    run_learning_loop(args.pdf, n_iterations=args.iterations, out_path=args.out)