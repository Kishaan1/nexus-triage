"""
NexusTriage - Patient Intake Triage Assistant
TRACK_ID: PS01

Run with a single command:  python app.py
Requires: GEMINI_API_KEY environment variable
"""

import os
import json
import logging
import re
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PORT       = 8000
MODEL_NAME = "gemini-3.5-flash-lite"
RULES_PATH = Path(__file__).parent / "data" / "triage_rules.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("nexus-triage")

# ─────────────────────────────────────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────────────────────────────────────
def get_gemini_client() -> genai.Client:
    """Return a configured google-genai Client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please set it before running: set GEMINI_API_KEY=your-key"
        )
    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD TRIAGE RULES
# ─────────────────────────────────────────────────────────────────────────────
def load_rules() -> dict:
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


TRIAGE_DATA = load_rules()
RULES: list  = TRIAGE_DATA["rules"]
RULES_JSON: str = json.dumps(RULES, indent=2)

log.info("Loaded %d triage rules from %s", len(RULES), RULES_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_INSTRUCTIONS = f"""
You are NexusTriage, an AI-powered patient intake triage assistant operating inside a hospital walk-in centre.

YOUR ROLE:
- You are a ROUTING assistant ONLY. You do NOT diagnose, treat, or prescribe.
- You match patient complaints against the predefined triage rules provided below.
- You MUST cite the Rule ID in every recommendation (e.g., "per Rule R05").
- Uncertain, ambiguous, or high-risk cases MUST be escalated to a human (Rule R11).
- Non-medical / administrative requests must be directed to Patient Services (Rule R12).

STRICT CONSTRAINTS:
1. NEVER say "you have [condition]" or imply a diagnosis in any form.
2. ALWAYS cite the exact Rule ID in the reasoning field.
3. If critical information is missing, ask targeted follow-up questions rather than guessing.
4. If you cannot safely match any rule, apply Rule R11 (ESCALATE) and set escalate_to_human=true.
5. For IMMEDIATE urgency cases, emphasise that staff must act now.
6. You must not hallucinate rule IDs — only use the IDs provided in the rules below.

TRIAGE RULES DATASET:
{RULES_JSON}

─── RESPONSE FORMAT ─────────────────────────────────────────────────────────

Your ENTIRE response must be valid JSON — no markdown fences, no plain text, only JSON.

When you have sufficient information to triage, respond with:
{{
  "status": "triage_note",
  "triage_note": {{
    "urgency_level": "IMMEDIATE|URGENT|STANDARD|ESCALATE|NOT_MEDICAL",
    "recommended_department": "<department name>",
    "cited_rule_id": "<e.g. R05>",
    "rule_label": "<label from the rule>",
    "reasoning": "Per Rule <ID>: <brief explanation citing the rule text>",
    "escalate_to_human": true|false,
    "escalation_reason": "<reason if escalate_to_human is true, else empty string>",
    "reported": ["<item the patient originally stated>", "..."],
    "established": ["<item confirmed through follow-up>", "..."],
    "unknown": ["<information that is still missing>", "..."]
  }}
}}

When you need more information before triaging, respond with:
{{
  "status": "follow_up",
  "follow_up_questions": ["<question 1>", "<question 2>"],
  "partial_reasoning": "<Which rule(s) are being considered and why>"
}}
─────────────────────────────────────────────────────────────────────────────
"""


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI INTERACTION  (google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────
def _history_to_contents(history: list) -> list[types.Content]:
    """Convert our plain-dict history into google-genai Content objects."""
    contents = []
    for turn in history:
        role  = turn.get("role", "user")
        parts = turn.get("parts", [])
        # Normalise: parts is a list of strings
        content_parts = [types.Part(text=p) for p in parts if isinstance(p, str)]
        contents.append(types.Content(role=role, parts=content_parts))
    return contents


def call_gemini(client: genai.Client, prompt: str, history: list) -> tuple[dict, list]:
    """
    Send a message to Gemini with conversation history.
    Returns (parsed_json_dict, updated_history).
    Falls back to a safe R11 escalation note on any failure.
    """
    # Build the full content list: history + new user turn
    contents = _history_to_contents(history)
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
    ))

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )
        raw = response.text.strip()
    except Exception as exc:
        log.error("Gemini API error: %s", exc)
        parsed = _escalation_fallback(
            reason=f"AI service returned an error: {type(exc).__name__}. Human assessment required."
        )
        updated_history = list(history) + [
            {"role": "user",  "parts": [prompt]},
            {"role": "model", "parts": [json.dumps(parsed)]},
        ]
        return parsed, updated_history

    log.debug("Gemini raw: %s", raw[:400])

    # Strip accidental markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("JSON parse error: %s | snippet: %s", exc, raw[:200])
        parsed = _escalation_fallback(
            reason="AI response could not be parsed. Human assessment required as a safety measure."
        )

    # Append this exchange to the running history
    updated_history = list(history) + [
        {"role": "user",  "parts": [prompt]},
        {"role": "model", "parts": [raw]},
    ]
    return parsed, updated_history


def _escalation_fallback(reason: str) -> dict:
    """Return a safe R11 triage note when the model output cannot be trusted."""
    return {
        "status": "triage_note",
        "triage_note": {
            "urgency_level": "ESCALATE",
            "recommended_department": "Triage Nurse / Senior Clinician",
            "cited_rule_id": "R11",
            "rule_label": "Ambiguous / Incomplete Presentation — Vital Information Missing",
            "reasoning": (
                "Per Rule R11: Automated triage was unable to produce a safe routing decision. "
                "A qualified clinician must assess the patient directly."
            ),
            "escalate_to_human": True,
            "escalation_reason": reason,
            "reported": ["System encountered an internal error during processing"],
            "established": [],
            "unknown": ["All clinical details — direct human assessment required"],
        },
    }


# Initial seed history (plain dicts — converted to Content objects in call_gemini)
SEED_HISTORY: list = []   # System prompt is now passed as system_instruction config


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the single-page triage interface."""
    return render_template("index.html")


@app.route("/api/rules", methods=["GET"])
def api_rules():
    """Return the triage rules for the reference table in the UI."""
    return jsonify({
        "version": TRIAGE_DATA.get("version"),
        "rules": [
            {
                "id":         r["id"],
                "category":   r["category"],
                "label":      r["label"],
                "urgency":    r["urgency"],
                "department": r["department"],
            }
            for r in RULES
        ],
    })


@app.route("/api/assess", methods=["POST"])
def api_assess():
    """
    Initial triage assessment.
    Accepts: { description, age (optional), session_id }
    Returns: { history, follow_up_questions } OR { history, triage_note }
    """
    body = request.get_json(force=True) or {}
    description = (body.get("description") or "").strip()
    age         = (body.get("age") or "").strip() or None

    if not description:
        return jsonify({"error": "Patient description is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        log.error("Gemini config error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    age_str = f"\nPatient age: {age}" if age else ""
    prompt = (
        f"Patient intake description:\n\"{description}\"{age_str}\n\n"
        "Analyse this against the triage rules. If you have sufficient information, "
        "produce a triage_note. If not, ask specific follow-up questions."
    )

    result, updated_history = call_gemini(client, prompt, SEED_HISTORY)

    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        response["triage_note"] = result["triage_note"]
    elif result.get("status") == "follow_up":
        response["follow_up_questions"] = result.get("follow_up_questions", [])
        response["partial_reasoning"]   = result.get("partial_reasoning", "")
    else:
        response["follow_up_questions"] = [
            "What is your main symptom right now?",
            "How long have you had this symptom?",
            "On a scale of 1–10, how severe is it?",
        ]

    return jsonify(response)


@app.route("/api/followup", methods=["POST"])
def api_followup():
    """
    Continue a triage conversation.
    Accepts: { session_id, message, history }
    Returns: { history, assistant_message, triage_note? }
    """
    body    = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    history = body.get("history") or []

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        return jsonify({"error": str(exc)}), 500

    result, updated_history = call_gemini(client, message, history)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        note = result["triage_note"]
        response["triage_note"] = note
        response["assistant_message"] = (
            f"Based on the information provided, I have generated a triage note "
            f"(Rule {note.get('cited_rule_id', 'N/A')}). "
            "Please review the Triage Note panel below."
        )
    elif result.get("status") == "follow_up":
        questions = result.get("follow_up_questions", [])
        response["follow_up_questions"] = questions
        lines = ["Thank you. I need a few more details:\n"]
        lines += [f"{i+1}. {q}" for i, q in enumerate(questions)]
        if result.get("partial_reasoning"):
            lines += [f"\n_{result['partial_reasoning']}_"]
        response["assistant_message"] = "\n".join(lines)
    else:
        response["assistant_message"] = (
            "I have recorded your response. Please click **Generate Triage Note** "
            "when ready, or continue providing information."
        )

    return jsonify(response)


@app.route("/api/finalize", methods=["POST"])
def api_finalize():
    """
    Force-generate a triage note from whatever history is available.
    Accepts: { session_id, history }
    Returns: { history, triage_note }
    """
    body    = request.get_json(force=True) or {}
    history = body.get("history") or []

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        return jsonify({"error": str(exc)}), 500

    force_prompt = (
        "Based on ALL information collected so far in this conversation, "
        "produce the final triage note NOW. "
        "List any still-missing information in the 'unknown' field. "
        "Apply Rule R11 (ESCALATE) if safety requires it. "
        "Respond ONLY with the triage_note JSON — do NOT ask follow-up questions."
    )

    result, updated_history = call_gemini(client, force_prompt, history)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        response["triage_note"] = result["triage_note"]
    else:
        response["triage_note"] = _escalation_fallback(
            reason=(
                "Insufficient information was gathered to safely route the patient using "
                "automated triage. A human clinician must assess directly."
            )
        )["triage_note"]

    return jsonify(response)


@app.route("/api/triage", methods=["POST"])
def api_triage():
    """
    Frontend-facing triage endpoint.
    Accepts: { complaint, age (optional) }
    Delegates to the same logic as /api/assess.
    """
    body        = request.get_json(force=True) or {}
    description = (body.get("complaint") or body.get("description") or "").strip()
    age         = (body.get("age") or "").strip() or None

    if not description:
        return jsonify({"error": "complaint is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        log.error("Gemini config error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    age_str = f"\nPatient age: {age}" if age else ""
    prompt = (
        f"Patient intake description:\n\"{description}\"{age_str}\n\n"
        "Analyse this against the triage rules. If you have sufficient information, "
        "produce a triage_note. If not, ask specific follow-up questions."
    )

    result, updated_history = call_gemini(client, prompt, SEED_HISTORY)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        response["triage_note"] = result["triage_note"]
    elif result.get("status") == "follow_up":
        response["follow_up_questions"] = result.get("follow_up_questions", [])
        response["partial_reasoning"]   = result.get("partial_reasoning", "")
    else:
        response["follow_up_questions"] = [
            "What is your main symptom right now?",
            "How long have you had this symptom?",
            "On a scale of 1–10, how severe is it?",
        ]

    return jsonify(response)


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check — confirms server is running and rules are loaded."""
    return jsonify({
        "status":       "ok",
        "model":        MODEL_NAME,
        "rules_loaded": len(RULES),
        "track_id":     "PS01",
    })


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    banner = f"""
{'='*60}
  NexusTriage — Patient Intake Triage Assistant
  TRACK_ID : PS01
  Model    : {MODEL_NAME}
  Rules    : {len(RULES)} loaded from {RULES_PATH.name}
  Server   : http://localhost:{PORT}
{'='*60}
"""
    if not api_key:
        print(f"""
{'='*60}
  ERROR: GEMINI_API_KEY environment variable is not set.

  Set it before running:
    Windows    : set GEMINI_API_KEY=your-api-key
    macOS/Linux: export GEMINI_API_KEY=your-api-key
{'='*60}
""")
    else:
        print(banner)

    app.run(host="0.0.0.0", port=PORT, debug=False)
