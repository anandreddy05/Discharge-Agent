"""
memory.py

Stores doctor corrections as a JSON list.
Retrieval is keyword-based (no embeddings): for each section,
find past corrections whose agent_value shares keywords with the current value.

Schema of each memory entry:
{
    "section":       str,          # e.g. "follow_up_instructions"
    "agent_value":   str,          # what the agent originally wrote
    "doctor_value":  str,          # what the doctor corrected it to
    "reward_before": float,        # section score before correction
    "patient_id":    str,          # for tracing
    "timestamp":     str           # ISO
}
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any

MEMORY_PATH = os.path.join(os.getcwd(), "correction_memory.json")
TOP_K = 3  # how many examples to inject per section


def _load() -> List[Dict]:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(memories: List[Dict]) -> None:
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)


def _serialise(val: Any) -> str:
    if isinstance(val, (dict, list)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def store_corrections(
    agent_draft: dict,
    doc_draft: dict,
    section_scores: Dict[str, float],
    patient_id: str,
) -> int:
    """
    Diff agent_draft vs doctor_draft section by section.
    Only store entries where the doctor actually changed something (score < 1.0).
    Returns number of new entries added.
    """
    memories = _load()
    added = 0
    ts = datetime.now(timezone.utc).isoformat()

    for section, score in section_scores.items():
        if section == "overall":
            continue
        if score >= 1.0:
            continue  # no correction needed 
            
        agent_val = agent_draft.get(section, "")
        doctor_val = doc_draft.get(section, "")
 

        if str(agent_val) == str(doctor_val):
            continue

        memories.append({
            "section": section,
            "agent_value": _serialise(agent_val),
            "doctor_value": _serialise(doctor_val),
            "reward_before": score,
            "patient_id": patient_id,
            "timestamp": ts,
        })
        added += 1
    _save(memories)
    return added

def _keywords(text: str) -> set:
    """Lowercase words longer than 3 chars."""
    return {w.lower() for w in text.replace(",", " ").split() if len(w) > 3}

def retrieve_for_section(
    section: str, 
    current_agent_value: Any, 
    top_k: int = TOP_K) -> List[Dict]:
    
    """
    Returns up to top_k past corrections for this section, ranked by keyword overlap
    with current_agent_value.
    """
    memories = _load()
    candidates = [m for m in memories if m["section"] == section]
 
    if not candidates:
        return []
 
    query_kw = _keywords(_serialise(current_agent_value))
 
    def overlap(entry: Dict) -> int:
        entry_kw = _keywords(entry["agent_value"])
        return len(query_kw & entry_kw)
 
    ranked = sorted(candidates, key=overlap, reverse=True)
    return ranked[:top_k]


def retrieve_all_sections(agent_draft: dict) -> Dict[str, List[Dict]]:
    """
    For every section in the draft, retrieve relevant past corrections.
    Returns {section: [memory_entry, ...]}
    """
    result = {}
    for section, value in agent_draft.items():
        hits = retrieve_for_section(section, value)
        if hits:
            result[section] = hits
    return result


 
def format_memory_for_prompt(agent_draft: dict) -> str:
    """
    Returns a formatted string ready to be injected into the reasoner system prompt.
    Empty string if no memories exist yet.
    """
    relevant = retrieve_all_sections(agent_draft)
    if not relevant:
        return ""
    
    lines = [
        "\n=== CORRECTION MEMORY (learn from past doctor edits) ===",
        "The following are real corrections a doctor made to previous drafts.",
        "Use them as few-shot examples to produce a better draft this time.\n",
    ]
    for section, entries in relevant.items():
        lines.append(f"[{section.upper()}]")
        for e in entries:
            lines.append(f"  Agent wrote : {e['agent_value']}")
            lines.append(f"  Doctor fixed: {e['doctor_value']}")
            lines.append(f"  (Score was {e['reward_before']:.2f} before correction)")
            lines.append("")
    lines.append("=== END CORRECTION MEMORY ===\n")
    return "\n".join(lines)


def memory_stats() -> Dict:
    memories = _load()
    if not memories:
        return {"total_entries": 0}
    
    by_section: Dict[str, List[float]] = {}
    for m in memories:
        by_section.setdefault(m["section"], []).append(m["reward_before"])
    
    return {
        "total_entries": len(memories),
        "by_section": {
            sec: {
                "count": len(scores),
                "avg_score_before_correction": round(sum(scores) / len(scores), 4),
            }
            for sec, scores in by_section.items()
        },
    }


