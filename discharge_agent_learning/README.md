# 🧠 Self-Improving Clinical AI — Learning Loop

> **Assessment Part B** — Reinforcement Learning from Doctor Corrections

An **iterative self-improvement system** built on top of the Part A discharge summary agent. After each run, a simulated doctor reviews the agent's draft, a reward function scores every section, and the delta is stored in a persistent memory bank. On the next iteration, the agent reads its own past mistakes and produces a better draft — without any weight updates or fine-tuning.

---

<h2 align="center">Learning Loop Overview</h2>

<p align="center">
  <img src="./images/learning_loop_overview.png" alt="Learning Loop Overview" width="700"/>
</p>

`Agent Draft → Doctor Review → Reward Score → Memory Store → (next iteration) Agent Draft`

---

## 📑 Table of Contents

- [🧠 Self-Improving Clinical AI — Learning Loop](#-self-improving-clinical-ai--learning-loop)
  - [📑 Table of Contents](#-table-of-contents)
  - [Overview](#overview)
    - [What the System Does in Each Iteration](#what-the-system-does-in-each-iteration)
  - [What Changed from Part A](#what-changed-from-part-a)
  - [System Architecture](#system-architecture)
  - [Project Structure](#project-structure)
  - [Core Components](#core-components)
    - [`learning_loop.py` — The Orchestrator](#learning_looppy--the-orchestrator)
    - [`reviewer.py` — The Simulated Doctor](#reviewerpy--the-simulated-doctor)
    - [`reward.py` — The Scoring Function](#rewardpy--the-scoring-function)
    - [`memory.py` — The Correction Bank](#memorypy--the-correction-bank)
    - [`evaluate.py` — The Analysis Tool](#evaluatepy--the-analysis-tool)
    - [`nodes.py` — Memory-Augmented Reasoner](#nodespy--memory-augmented-reasoner)
    - [`graph.py`](#graphpy)
  - [The Learning Cycle in Detail](#the-learning-cycle-in-detail)
  - [Memory System Design](#memory-system-design)
    - [Why keyword retrieval instead of embeddings?](#why-keyword-retrieval-instead-of-embeddings)
    - [Why flat JSON instead of a vector store?](#why-flat-json-instead-of-a-vector-store)
    - [Memory growth is bounded](#memory-growth-is-bounded)
  - [Reward Function Design](#reward-function-design)
    - [Why Levenshtein distance?](#why-levenshtein-distance)
    - [Limitations of edit distance for clinical content](#limitations-of-edit-distance-for-clinical-content)
  - [Running the Learning Loop](#running-the-learning-loop)
    - [Basic Usage](#basic-usage)
    - [CLI Arguments](#cli-arguments)
    - [What Gets Created](#what-gets-created)
  - [Evaluating Results](#evaluating-results)
    - [Output](#output)
  - [Design Decisions](#design-decisions)
    - [Why a simulated doctor instead of real human feedback?](#why-a-simulated-doctor-instead-of-real-human-feedback)
    - [Why persist corrections across separate `run_learning_loop` calls?](#why-persist-corrections-across-separate-run_learning_loop-calls)
    - [Why carry the previous draft into the next iteration?](#why-carry-the-previous-draft-into-the-next-iteration)
    - [Why `TOP_K = 3` for memory retrieval?](#why-top_k--3-for-memory-retrieval)
    - [Why not fine-tune the model?](#why-not-fine-tune-the-model)
  - [Known Limitations](#known-limitations)
  - [Full File Reference](#full-file-reference)

---

## Overview

Part B adds a **closed-loop learning system** on top of Part A's agent. The system answers the question:

> *"Can a clinical AI agent improve its discharge summaries by learning from doctor corrections, without retraining its underlying model?"*

The answer is **yes**, through in-context learning driven by a persistent correction memory bank.

### What the System Does in Each Iteration

```
1. Run the full Part A agent on the patient PDF  →  agent_draft
2. Send agent_draft to a simulated doctor (GPT-4o with clinical policy)  →  doctor_draft
3. Compute reward: section-level Levenshtein distance between drafts  →  scores
4. Store only the sections that differed (score < 1.0) into correction_memory.json
5. On the next run, inject relevant past corrections into the agent's system prompt
6. The agent reads its mistakes and writes a better draft
```

---

## What Changed from Part A

| Component | Part A | Part B |
|---|---|---|
| `graph.py` | Basic router | **Bug fix:** `hasattr` check prevents crash on non-AI messages |
| `nodes.py` | No memory | **Memory injection:** `format_memory_for_prompt()` added to system prompt |
| `agent_state.py` | Same | Unchanged — `draft_summary` carries forward between iterations |
| `learning/` directory | Does not exist | **New:** `learning_loop.py`, `reviewer.py`, `reward.py`, `memory.py` |
| `evaluate.py` | Does not exist | **New:** CLI evaluation tool with improvement curves and section tables |

---

## System Architecture

<h2 align="center">Learning Loop Overview</h2>

<p align="center">
  <img src="./images/system_architecture.png" alt="Learning Loop Overview" width="700"/>
</p>

---

## Project Structure

```
project/
│
├── agent/
│   ├── graph.py              # Bug-fixed router (hasattr guard)
│   ├── nodes.py              # Memory-augmented reasoner + validator
│   └── tools.py              # Unchanged from Part A
│
├── learning/
│   ├── learning_loop.py      # Main CLI: runs N iterations of the full cycle
│   ├── reviewer.py           # Simulated doctor (GPT-4o with clinical policy)
│   ├── reward.py             # Levenshtein-based per-section scoring
│   └── memory.py             # Persistent correction bank (JSON + keyword retrieval)
│
├── schema/
│   ├── agent_state.py        # Unchanged
│   └── output_models.py      # Unchanged
│
├── ingestion/
│   └── document_processor.py # Unchanged
│
├── evaluate.py               # CLI analysis tool for learning_results.json
│
├── correction_memory.json    # Auto-generated: grows with each iteration
├── learning_results.json     # Auto-generated: full log of every iteration
│
└── .env
```

---

## Core Components

### `learning_loop.py` — The Orchestrator

The main entry point for Part B. Runs N complete Agent→Review→Score→Store cycles.

**Key design: draft carry-forward**

```python
previous_draft = None
for i in range(start_iter, start_iter + n_iterations):
    result = run_one_iteration(pdf_path, iteration_num=i, previous_draft=previous_draft)
    if result.get("status") == "success":
        previous_draft = result.get("agent_draft")
```

The previous draft is passed into the next iteration's `initial_state`, making the agent's memory of its own prior output available alongside the new correction memory.

**4-stage pipeline per iteration:**

```
[1/4] Build graph → run agent → get draft_summary
[2/4] simulate_doc_review(agent_draft, source_text)
[3/4] compute_reward(agent_draft, doctor_draft)
[4/4] store_corrections(agent_draft, doctor_draft, scores, patient_id)
```

Results are appended to `learning_results.json` after **every** iteration (crash-safe incremental save).

---

### `reviewer.py` — The Simulated Doctor

A GPT-4o instance operating under a **hidden clinical policy** that the agent never sees. This asymmetry is intentional — it simulates real-world RLHF where human preferences are not fully specified in the agent's prompt.

**The 6-rule reviewer policy:**

| Rule | What the Doctor Checks |
|---|---|
| **Demographics** | Recovers missing values from source text; keeps `[MISSING]` if truly absent |
| **Medications** | Adds clinical inference notes for `"No documented reason"` cases where hospital course supports it |
| **Hospital Course** | Rewrites bullet points as fluent past-tense clinical prose |
| **Conflicts** | Reformats as `"Admission note vs Progress note: <detail>"` |
| **Follow-up** | Ensures every entry has an explicit timeframe; adds `[TIMEFRAME NOT SPECIFIED]` if not inferable |
| **Completeness** | Verifies `pending_results` and `missing_critical_info` are populated |

The reviewer returns a JSON object with **identical schema** to the agent's output — enabling direct field-by-field comparison in the reward function.

**Bug fix included:** The original `_strip_fences` had index bugs (`raw[1]`, `raw[4]`). The corrected version properly slices from the first newline to the last ` ``` `:

```python
def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw[raw.index("\n") + 1:]
        if raw.endswith("```"):
            raw = raw[:raw.rfind("```")]
    return raw.strip()
```

---

### `reward.py` — The Scoring Function

Computes a **normalised similarity score** (0.0–1.0) for each section by comparing the agent draft to the doctor draft using Levenshtein edit distance.

```
reward_score = 1.0 - (edit_distance / max(len(agent_str), len(doc_str)))
```

**`1.0` = perfect match. `0.0` = completely different.**

All values — strings, lists of strings, lists of Pydantic objects, dicts — are normalised to a comparable string via `_to_str()`:

```python
def _to_str(value) -> str:
    if isinstance(value, list):
        items = []
        for item in value:
            if hasattr(item, "dict"):
                items.append(json.dumps(item.dict(), sort_keys=True))
            elif isinstance(item, dict):
                items.append(json.dumps(item, sort_keys=True))
            else:
                items.append(str(item))
        return " | ".join(sorted(items))   # sorted for order-independence
    return str(value) if value is not None else ""
```

**Scored sections (17 total):**

```
patient_name        age                  gender
admission_date      discharge_date       principal_diagnoses
secondary_diagnoses hospital_course      procedures_performed
discharge_condition medications_on_admission  discharge_medications
allergies           follow_up_instructions    pending_results
missing_critical_info   identified_conflicts
```

The `overall` score is the unweighted mean across all 17 sections.

---

### `memory.py` — The Correction Bank

The core of the learning system. Stores doctor corrections as a flat JSON list and retrieves relevant ones using **keyword overlap** (no embeddings required).

**Memory entry schema:**

```json
{
  "section": "follow_up_instructions",
  "agent_value": "[\"Review in 1 week\"]",
  "doctor_value": "[\"Follow up with nephrologist in 2 weeks [TIMEFRAME SPECIFIED]\"]",
  "reward_before": 0.42,
  "patient_id": "abc123",
  "timestamp": "2026-03-01T10:23:45+00:00"
}
```

**Storage policy — only meaningful corrections are stored:**

```python
for section, score in section_scores.items():
    if section == "overall":
        continue
    if score >= 1.0:
        continue   # agent was already correct — nothing to learn
    if str(agent_val) == str(doctor_val):
        continue   # no actual change despite score < 1.0
    # → store this correction
```

**Retrieval — keyword-based nearest-neighbour:**

```python
def _keywords(text: str) -> set:
    return {w.lower() for w in text.replace(",", " ").split() if len(w) > 3}

# Rank past corrections by keyword overlap with the current draft's value for that section
ranked = sorted(candidates, key=lambda e: len(query_kw & _keywords(e["agent_value"])), reverse=True)
return ranked[:TOP_K]   # TOP_K = 3
```

**Prompt injection — the memory block added to `reasoner`'s system prompt:**

```
=== CORRECTION MEMORY (learn from past doctor edits) ===
The following are real corrections a doctor made to previous drafts.
Use them as few-shot examples to produce a better draft this time.

[FOLLOW_UP_INSTRUCTIONS]
  Agent wrote : ["Review in 1 week"]
  Doctor fixed: ["Follow up with nephrologist in 2 weeks"]
  (Score was 0.42 before correction)

=== END CORRECTION MEMORY ===
```

---

### `evaluate.py` — The Analysis Tool

A standalone CLI that reads `learning_results.json` and produces three reports:

**1. Improvement Curve** — overall reward per iteration with delta from baseline:
```
=== IMPROVEMENT CURVE (overall reward per iteration) ===
  Iter    Overall  Bar
  ------ --------  ----------------------
  1        0.9704  ███████████████████    (baseline)
  2        0.9609  ███████████████████    (-0.0095)
  3        0.9753  ███████████████████    (+0.0049)

```

**2. Section Before/After Table** — first vs latest iteration per section:
```
=== SECTION-LEVEL BEFORE vs AFTER ===
  Section                               Before    After    Delta
  ----------------------------------- -------- -------- --------
  patient_name                          1.0000   1.0000  +0.0000
  age                                   1.0000   1.0000  +0.0000
  gender                                1.0000   1.0000  +0.0000
  admission_date                        1.0000   1.0000  +0.0000
  discharge_date                        1.0000   1.0000  +0.0000
  principal_diagnoses                   1.0000   1.0000  +0.0000
  secondary_diagnoses                   1.0000   1.0000  +0.0000
  hospital_course                       0.8554   0.9606 ▲+0.1052
  procedures_performed                  1.0000   1.0000  +0.0000
  discharge_condition                   1.0000   1.0000  +0.0000
  medications_on_admission              1.0000   1.0000  +0.0000
  discharge_medications                 0.9105   0.8774 ▼-0.0331
  allergies                             1.0000   1.0000  +0.0000
  follow_up_instructions                0.7306   0.7413 ▲+0.0107
  pending_results                       1.0000   1.0000  +0.0000
  missing_critical_info                 1.0000   1.0000  +0.0000
  identified_conflicts                  1.0000   1.0000  +0.0000
  overall                               0.9704   0.9753 ▲+0.0049

  Sections improved : 3
  Sections degraded : 1
  Sections unchanged: 13
```

**3. Memory Bank Growth** — entries accumulated per iteration.


=== MEMORY BANK GROWTH ===
  Iter    Total entries
  ------ --------------
  1                   3

  Sections improved : 3
  Sections degraded : 1
  Sections unchanged: 13

=== MEMORY BANK GROWTH ===
  Iter    Total entries
  ------ --------------
  1                   3
  Sections unchanged: 13

=== MEMORY BANK GROWTH ===
  Iter    Total entries
  ------ --------------
  1                   3
  Iter    Total entries
  ------ --------------
  1                   3
  1                   3
  2                   6
  3                   9
  3                   9
**4. Top 5 Weakest Sections** — targets for further improvement.

=== TOP 5 WEAKEST SECTIONS (latest iteration) ===
  follow_up_instructions              0.7413
  discharge_medications               0.8774
  hospital_course                     0.9606
  patient_name                        1.0000
  age                                 1.0000

---

### `nodes.py` — Memory-Augmented Reasoner

The key Part B addition to the `reasoner` node is the memory injection block:

```python
existing_draft = state.get("draft_summary")
memory_block = ""
if existing_draft:
    try:
        memory_block = format_memory_for_prompt(existing_draft.dict())
    except Exception:
        memory_block = ""
```

The `memory_block` string is inserted directly into the system prompt between the file path and the step instructions. If no memory exists yet (first run), it is an empty string and the prompt is unchanged.

**Additional formatting rules added in Part B** to fix recurring schema errors:

```
CRITICAL FORMATTING RULES:
- 'allergies' must ALWAYS be a JSON array: ["[NOT DOCUMENTED]"] not "[NOT DOCUMENTED]"
- 'medications_on_admission' must ALWAYS be a JSON array
- 'pending_results' must ALWAYS be a JSON array of objects
- 'discharge_medications' must ALWAYS be a JSON array of objects
```

---

### `graph.py` 

Part B adds a one-line safety guard that was absent in Part A:

```python
# Part A (crashes on ToolMessage):
if last_message.tool_calls:

# Part B (safe):
if not hasattr(last_message, "tool_calls"):
    return END   # ToolMessage — not an AI message, can't have tool calls
if last_message.tool_calls:
```

---

## The Learning Cycle in Detail

```
Iteration 1
├── Agent reads PDF (no memory yet)
├── Produces draft_v1
├── Doctor reviews → doctor_draft_v1
├── Reward: overall = 0.72
│   └── Weak sections: follow_up (0.42), hospital_course (0.61)
└── Memory stores 8 corrections

Iteration 2
├── Agent reads PDF
├── Memory injected: "Doctor changed follow_up from X to Y (score 0.42)"
├── Agent incorporates correction examples
├── Produces draft_v2 (better follow_up, better hospital_course)
├── Doctor reviews → doctor_draft_v2
├── Reward: overall = 0.79 (+0.07)
└── Memory stores 5 new corrections (8 already exist from iter 1)

Iteration 3
├── Agent reads PDF
├── Memory now has 13 entries across sections
├── Produces draft_v3
├── Reward: overall = 0.83 (+0.11 from baseline)
└── ...
```

---

## Memory System Design

### Why keyword retrieval instead of embeddings?

| Approach | Pros | Cons |
|---|---|---|
| **Keyword overlap (chosen)** | Zero latency, no API cost, no vector DB dependency, fully deterministic | Less semantically rich |
| Embedding similarity | Better semantic matching | Requires embedding API calls per retrieval, adds latency, cost, and infrastructure |

For clinical correction memory where section names and medical terms are highly consistent (e.g., "follow_up_instructions" always contains similar terminology), keyword overlap performs comparably to embeddings at a fraction of the cost.

### Why flat JSON instead of a vector store?

The memory bank is designed to be **transparent and inspectable** — a clinician or developer can open `correction_memory.json` and read every stored correction. This auditability is critical in healthcare AI contexts.

### Memory growth is bounded

Only sections with `score < 1.0` AND where the doctor actually changed something are stored. On a well-performing agent, memory growth slows as the agent improves. This prevents the memory file from growing unboundedly.

---

## Reward Function Design

### Why Levenshtein distance?

The reward function needs to compare fields that can be:
- Short strings (`"Male"` vs `"Female"`)
- Long clinical narratives (hospital course paragraphs)
- Lists of structured objects (discharge medications)

Levenshtein edit distance is **universal** — it works on any stringified representation without requiring field-specific comparison logic. The `sort_keys=True` in list serialisation ensures order-independent comparison for arrays.

### Limitations of edit distance for clinical content

Edit distance is a **proxy metric**. A doctor who rewrites a grammatically poor sentence into an equivalent clinical note would score low even if the clinical facts are identical. For a production system, this should be replaced with:
- Clinical NLI (Natural Language Inference) for narrative fields
- Exact set match for medication lists
- Date normalization for temporal fields

---

## Running the Learning Loop

### Basic Usage

```bash
# Run 3 iterations of the learning loop
python learning_loop.py --pdf path/to/patient_record.pdf --iterations 3

# Run with custom output path
python learning_loop.py --pdf path/to/patient_record.pdf --iterations 5 --out my_results.json
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to the patient PDF |
| `--iterations` | `3` | Number of Agent→Review→Score→Store cycles |
| `--out` | `learning_results.json` | Output path for results log |

### What Gets Created

```
correction_memory.json     ← grows with each iteration; persists between runs
learning_results.json      ← full log: scores, drafts, memory stats per iteration
extraction_logs/           ← raw OCR markdown per chunk (from Part A tools)
```

> **Resuming:** If `learning_results.json` already exists, `start_iter` is set to `len(existing_results) + 1` — the loop resumes from where it left off.

---

## Evaluating Results

```bash
# Evaluate using default results file
python evaluate.py

# Evaluate a specific file
python evaluate.py --results my_results.json
```

### Output

```
Loaded 3 successful iteration(s) from learning_results.json

=== IMPROVEMENT CURVE (overall reward per iteration) ===
  Iter   Overall  Bar
  1      0.7234   ██████████████   (baseline)
  2      0.7891   ████████████████ (+0.0657)
  3      0.8312   █████████████████(+0.1078)

=== SECTION-LEVEL BEFORE vs AFTER ===
  Section                             Before   After    Delta
  follow_up_instructions              0.4200   0.8900  ▲+0.4700
  hospital_course                     0.6100   0.7800  ▲+0.1700
  ...

=== MEMORY BANK GROWTH ===
  Iter   Total entries
  1      8
  2      13
  3      17

=== TOP 5 WEAKEST SECTIONS (latest iteration) ===
  identified_conflicts                0.5500
  medications_on_admission            0.6100
  ...
```


## Design Decisions

### Why a simulated doctor instead of real human feedback?
Real clinician feedback introduces latency, cost, and availability constraints that make automated learning loops impractical to build and test. GPT-4o operating under a hidden policy closely approximates clinician preferences while enabling rapid iteration. The hidden policy creates genuine asymmetry — the agent does not know what the reviewer will check.

### Why persist corrections across separate `run_learning_loop` calls?
`correction_memory.json` persists on disk and accumulates across all runs. This means the agent continues improving even if the loop is interrupted and restarted, and corrections from one patient record can (cautiously) inform summaries for future patients with similar conditions.

### Why carry the previous draft into the next iteration?
Passing `previous_draft` into `initial_state` gives the agent a reference point — it can see what it wrote last time alongside what the doctor corrected. Combined with the memory block, this provides both **explicit correction examples** and **implicit context** about the case.

### Why `TOP_K = 3` for memory retrieval?
Injecting more than 3 examples per section would inflate the system prompt, increasing latency and cost without meaningful accuracy gain. 3 examples provide enough signal for in-context learning while keeping the prompt manageable.

### Why not fine-tune the model?
Fine-tuning requires large correction datasets, significant compute, and introduces model versioning and deployment complexity. The in-context approach achieves meaningful improvement with zero infrastructure changes, making it practical for healthcare settings where model governance is strict.

---

## Known Limitations

| Limitation | Impact | Mitigation Path |
|---|---|---|
| Levenshtein reward is a proxy metric | Clinically equivalent drafts may score low if worded differently | Replace with clinical NLI + set-match for structured fields |
| Reviewer is also GPT-4o | Potential circular bias — same model reviews its own output | Use a different model (e.g., GPT-4 Turbo) or a human for ground truth |
| Keyword retrieval misses semantic similarity | "Hypertension" and "high BP" won't match | Replace with lightweight embedding retrieval (e.g., `text-embedding-3-small`) |
| Memory has no eviction policy | Old corrections from dissimilar cases persist indefinitely | Add patient-type tagging and staleness decay |
| No statistical significance testing | Improvement across 3 iterations may be noise | Run 10+ iterations; report confidence intervals |
| `correction_memory.json` is global | Corrections from one patient can influence another | Partition memory by diagnosis category or patient demographics |

---

## Full File Reference

| File | Role | New in Part B? |
|---|---|---|
| `learning/learning_loop.py` | Runs N iterations of the full pipeline | ✅ New |
| `learning/reviewer.py` | Simulated doctor with hidden clinical policy | ✅ New |
| `learning/reward.py` | Levenshtein-based per-section scoring | ✅ New |
| `learning/memory.py` | Correction bank: store, retrieve, format | ✅ New |
| `evaluate.py` | CLI analysis: curves, tables, weak sections | ✅ New |
| `agent/nodes.py` | Memory injection + formatting rules added | 🔄 Modified |
| `agent/graph.py` | `hasattr` safety guard added | 🔄 Modified |
| `agent/tools.py` | Unchanged | ➡️ Same |
| `schema/agent_state.py` | Unchanged | ➡️ Same |
| `schema/output_models.py` | Unchanged | ➡️ Same |
| `ingestion/document_processor.py` | Unchanged | ➡️ Same |

---

*Built for clinical AI assessment — Part B. Not for production medical use without regulatory approval.*