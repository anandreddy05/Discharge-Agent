import json
import os
from typing import Dict

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


REVIEWER_POLICY = """
You are a senior clinician reviewing an AI-generated discharge summary draft.
Apply the following editing policy CONSISTENTLY to every draft you receive:

POLICY (hidden from the AI agent, applied by you):
1. DEMOGRAPHICS: If any demographic field contains "[MISSING - FLAG FOR REVIEW]"
   but the source text contains the value, replace it with the correct value.
   If truly absent, keep the flag.

2. MEDICATIONS: For any medication that was Started/Stopped/Changed with
   "No documented reason", add a brief clinical inference note like
   "(likely due to <clinical context>)" — but ONLY if the hospital_course
   supports a clear inference. Otherwise keep the flag.

3. HOSPITAL COURSE: Rewrite as fluent past-tense clinical prose if it reads
   like bullet points or is incomplete. Preserve all facts.

4. CONFLICTS: Rephrase any identified_conflicts as:
   "Admission note vs Progress note: <detail of conflict>"

5. FOLLOW-UP: Ensure every follow_up_instructions entry includes a timeframe
   (e.g., "Follow up with cardiologist in 2 weeks"). Add "[TIMEFRAME NOT SPECIFIED]"
   if no timeframe can be inferred.

6. COMPLETENESS: Verify pending_results and missing_critical_info are populated.
   Do not invent clinical facts not present in the draft.

Return ONLY a valid JSON object matching the exact same schema as the input.
No preamble, no markdown fences.
"""


def _get_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_ADMIN_KEY")
    base_url = os.getenv("base_url")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        api_key=api_key,
        base_url=base_url,
    )


def _strip_fences(raw: str) -> str:
    """
    BUG FIX: original used raw[1] / raw[4] (single char index) instead of slicing.
    Properly strips ```json ... ``` or ``` ... ``` fences.
    """
    raw = raw.strip()
    if raw.startswith("```"):
        # Remove opening fence line
        raw = raw[raw.index("\n") + 1:]
        # Remove closing fence
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]
    return raw.strip()


def simulate_doc_review(agent_draft: dict, source_text: str = "") -> dict:
    """
    Takes the agent's draft dict and returns a doctor-corrected draft dict.
    source_text: raw extracted text from the PDF (passed in for demographic recovery).
    """
    llm = _get_llm()

    prompt = f"""{REVIEWER_POLICY}

SOURCE TEXT:
{source_text if source_text else "[No source text provided]"}

AGENT DRAFT TO REVIEW:
{json.dumps(agent_draft, indent=2)}

Return only the corrected JSON object."""

    response = llm.invoke([{"role": "user", "content": prompt}])
    raw = _strip_fences(response.content)

    try:
        corrected = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[REVIEWER WARNING] Failed to parse corrected draft: {e}")
        corrected = agent_draft  # safe fallback — return original

    return corrected