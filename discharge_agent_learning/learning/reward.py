from typing import Dict
import Levenshtein
import json

SCORED_SECTIONS = [
    "patient_name",
    "age",
    "gender",
    "admission_date",
    "discharge_date",
    "principal_diagnoses",
    "secondary_diagnoses",
    "hospital_course",
    "procedures_performed",
    "discharge_condition",
    "medications_on_admission",
    "discharge_medications",
    "allergies",
    "follow_up_instructions",
    "pending_results",
    "missing_critical_info",
    "identified_conflicts",
]


def _to_str(value) -> str:
    """Normalise any field value to a comparable string."""
    if isinstance(value, list):
        items = []
        for item in value:  
            if hasattr(item, "dict"):
                items.append(json.dumps(item.dict(), sort_keys=True))
            elif isinstance(item, dict):
                items.append(json.dumps(item, sort_keys=True))
            else:
                items.append(str(item))
        return " | ".join(sorted(items))
    return str(value) if value is not None else ""


def _edit_distance(agent_val, doc_val) -> float:
    """Normalised edit distance for a single section. Returns 0.0 = identical."""
    a = _to_str(agent_val)
    b = _to_str(doc_val)
    if not a and not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return Levenshtein.distance(a, b) / max_len


def compute_reward(agent_draft: dict, doc_draft: dict) -> Dict[str, float]:
    """
    Returns {section: reward_score} + overall.
    reward_score = 1.0 - normalised_edit_distance  (1.0 = perfect match)
    """
    scores = {}
    for section in SCORED_SECTIONS:
        agent_val = agent_draft.get(section, "")
        doc_val = doc_draft.get(section, "")
        dist = _edit_distance(agent_val, doc_val)
        scores[section] = round(1.0 - dist, 4)

    
    scores["overall"] = round(
        sum(scores[s] for s in SCORED_SECTIONS) / len(SCORED_SECTIONS), 4
    )
    return scores


def reward_summary(scores: Dict[str, float]) -> str:
    """Human-readable reward report."""
    lines = ["=== REWARD REPORT ==="]
    for section, score in scores.items():
        bar = "█" * int(score * 20)
        lines.append(f"  {section:<35} {score:.4f}  {bar}")
    return "\n".join(lines)
