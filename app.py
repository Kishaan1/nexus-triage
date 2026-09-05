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
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

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

AMBIGUITY & FOLLOW-UP PROTOCOL:
- When an initial patient complaint is ambiguous, unlocalized (such as vague abdominal or stomach pain without quadrant localization), or missing essential clinical discriminators (such as onset, fever, or severity), you MUST set status to "follow_up".
- Return 2 to 3 targeted clinical follow-up questions in "follow_up_questions" to clarify quadrant localization, onset, and red-flag symptoms.
- Do NOT immediately resolve an ambiguous initial intake to a closed Rule R11 triage note. Follow-up clarification MUST be attempted first.

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


AMBIGUOUS_TRIGGERS = [
    "not sure", "don't know", "dont know", "unclear", "vague", "maybe",
    "just feel bad", "something's wrong", "sometimes", "all over", "not certain"
]


def is_ambiguous_presentation(text: str, history: list = None) -> bool:
    """
    Check if the initial presentation has clinical ambiguity or lacks vital discriminators.
    Only triggers on initial assessment (when history is empty or has < 2 items).
    """
    if history and len(history) >= 2:
        return False

    t_lower = (text or "").lower()

    # If it is a clear emergency with unequivocal red flags, do not delay routing
    clear_emergencies = [
        "crushing chest pain", "spreads to my left arm", "radiating to arm",
        "can't breathe", "cannot breathe", "struggling to breathe", "gasping",
        "vomiting blood", "rigid abdomen", "heavy bleeding", "unconscious"
    ]
    for em in clear_emergencies:
        if em in t_lower:
            return False

    # Check for abdominal / stomach ambiguity
    has_abdominal = any(w in t_lower for w in ["stomach", "abdominal", "belly", "tummy", "abdomen"])
    has_ambiguity_flag = any(w in t_lower for w in AMBIGUOUS_TRIGGERS)

    if has_abdominal:
        has_quadrant = any(q in t_lower for q in [
            "right lower", "lower right", "rlq", "left lower", "lower left", "llq",
            "right upper", "upper right", "ruq", "left upper", "upper left", "luq",
            "epigastric", "groin", "flank"
        ])
        has_severe_signs = any(s in t_lower for s in [
            "rigid", "rebound", "vomiting blood", "blood in stool", "fever", "pregnant", "peritonitis"
        ])
        # If abdominal pain has ambiguous phrasing, or lacks both quadrant and severe signs
        if has_ambiguity_flag or (not has_quadrant and not has_severe_signs):
            return True

    # General ambiguity (e.g. Rule R11 triggers without definitive symptom pattern)
    if has_ambiguity_flag:
        return True

    return False


def generate_followup_response(text: str, history: list = None) -> dict:
    t_lower = (text or "").lower()
    has_abdominal = any(w in t_lower for w in ["stomach", "abdominal", "belly", "tummy", "abdomen"])

    if has_abdominal:
        questions = [
            "Where is the pain located right now (e.g., lower-right side, upper stomach, or diffuse across the whole abdomen)?",
            "Did the pain start suddenly, or has it been developing gradually over time?",
            "Are you experiencing any fever, chills, persistent vomiting, or inability to keep fluids down?",
        ]
        reasoning = (
            "Differentiating between Acute Abdominal Pain (Rule R09) and Mild Abdominal Discomfort (Rule R10). "
            "Clarification of quadrant localization, onset rapidity, and systemic danger signs is required before safe clinical routing."
        )
    else:
        questions = [
            "Can you describe your single most bothersome symptom right now?",
            "When did this symptom start, and was the onset sudden or gradual?",
            "Do you have any associated symptoms like fever, shortness of breath, or dizziness?",
        ]
        reasoning = (
            "Presentation lacks critical clinical discriminators (Rule R11 criteria). "
            "Targeted clarification is required to determine the appropriate clinical routing protocol."
        )

    return {
        "status": "follow_up",
        "requires_followup": True,
        "follow_up_questions": questions,
        "partial_reasoning": reasoning,
    }


def deterministic_rule_match(text: str, history: list = None) -> dict:
    """
    Deterministic local triage engine citing grounded rules dataset R01-R12.
    Used when Gemini API is offline, rate-limited, or validating grounded protocols.
    """
    if is_ambiguous_presentation(text, history):
        return generate_followup_response(text, history)

    full_text = text or ""
    if history:
        for turn in history:
            parts = turn.get("parts", [])
            for p in parts:
                if isinstance(p, str):
                    p_str = p.strip()
                    if p_str.startswith("{") and p_str.endswith("}"):
                        continue
                    full_text += " " + p

    full_lower = full_text.lower()
    best_rule = None
    best_score = 0

    for rule in RULES:
        score = 0
        cat = rule.get("category", "").lower().replace("_", " ")
        for kw in rule.get("trigger_keywords", []):
            if isinstance(kw, str):
                kw_l = kw.lower()
                if re.search(r"\b" + re.escape(kw_l) + r"\b", full_lower):
                    score += len(kw.split()) * 4
                elif any(w in full_lower for w in kw_l.split() if len(w) > 4):
                    score += 2
        for cond_key, cond_val in rule.get("conditions", {}).items():
            if isinstance(cond_val, list):
                for flag in cond_val:
                    if isinstance(flag, str):
                        fl = flag.lower()
                        if re.search(r"\b" + re.escape(fl) + r"\b", full_lower):
                            score += 5
                        elif any(re.search(r"\b" + re.escape(w) + r"\b", full_lower) for w in fl.split() if len(w) > 4):
                            score += 2
        if cat and (cat in full_lower or any(w in full_lower for w in cat.split() if len(w) > 4)):
            score += 2
        if score > best_score:
            best_score = score
            best_rule = rule

    if not best_rule or best_score == 0:
        best_rule = next((r for r in RULES if r["id"] == "R11"), RULES[10])

    user_inputs = []
    if history:
        for turn in history:
            if turn.get("role") == "user":
                for p in turn.get("parts", []):
                    if isinstance(p, str):
                        m = re.search(r'Patient intake description:\s*"([^"]+)"', p, re.DOTALL)
                        clean = m.group(1) if m else p
                        clean = re.sub(r"Based on ALL information collected.*", "", clean, flags=re.DOTALL).strip()
                        if clean:
                            user_inputs.append(clean)
    if text:
        m = re.search(r'Patient intake description:\s*"([^"]+)"', text, re.DOTALL)
        clean = m.group(1) if m else text
        clean = re.sub(r"Based on ALL information collected.*", "", clean, flags=re.DOTALL).strip()
        if clean and (not user_inputs or clean != user_inputs[-1]):
            user_inputs.append(clean)

    if len(user_inputs) > 1:
        reported = [s.strip() for s in user_inputs[0].replace("\n", " ").split(",") if s.strip()][:3]
        established = [s.strip() for s in user_inputs[-1].replace("\n", " ").split(",") if s.strip()][:3]
    else:
        raw = user_inputs[0] if user_inputs else text
        symptoms = [s.strip() for s in raw.replace("\n", " ").split(",") if s.strip()]
        reported = symptoms[:4] if symptoms else [raw[:120]]
        established = []

    unknown = best_rule.get("required_info", ["vital signs", "clinical examination"])

    return {
        "status": "triage_note",
        "requires_followup": False,
        "triage_note": {
            "urgency_level": best_rule.get("urgency", "ESCALATE"),
            "recommended_department": best_rule.get("department", "Emergency"),
            "cited_rule_id": best_rule.get("id", "R11"),
            "rule_label": best_rule.get("label", "Ambiguous Presentation"),
            "reasoning": f"Per Rule {best_rule['id']}: {best_rule.get('reasoning', '')}",
            "escalate_to_human": best_rule.get("escalate_to_human", True),
            "escalation_reason": "Escalated for immediate clinical safety evaluation" if best_rule.get("escalate_to_human") else "",
            "reported": reported,
            "established": established,
            "unknown": unknown,
        },
    }


def call_gemini(client: genai.Client | None, prompt: str, history: list) -> tuple[dict, list]:
    """
    Send a message to Gemini with conversation history.
    Returns (parsed_json_dict, updated_history).
    Falls back to deterministic local rule engine on any API failure.
    """
    if client is None:
        log.info("Gemini client is None — applying local deterministic triage rules")
        parsed = deterministic_rule_match(prompt, history)
        updated_history = list(history) + [
            {"role": "user",  "parts": [prompt]},
            {"role": "model", "parts": [json.dumps(parsed)]},
        ]
        return parsed, updated_history

    # Build the full content list: history + new user turn
    contents = _history_to_contents(history)
    contents.append(types.Content(
        role="user",
        parts=[types.Part(text=prompt)]
    ))

    raw = None
    candidate_models = [MODEL_NAME, "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-flash"]
    seen = set()
    models = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_exc = None
    for model_id in models:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTIONS,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
            raw = response.text.strip()
            if raw:
                log.info("Gemini response generated using model: %s", model_id)
                break
        except Exception as exc:
            last_exc = exc
            log.warning("Gemini model %s call failed: %s", model_id, exc)

    if not raw:
        log.warning("All Gemini model attempts failed (%s) — falling back to deterministic local rules", last_exc)
        parsed = deterministic_rule_match(prompt, history)
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

    # Post-process parsed response to enforce clarification protocol on ambiguous presentations:
    if parsed.get("status") == "follow_up":
        parsed["requires_followup"] = True
    elif parsed.get("status") == "triage_note":
        # If model immediately returned R11 on initial turn for an ambiguous complaint,
        # redirect it into the follow-up clarification protocol
        if (not history or len(history) == 0) and parsed.get("triage_note", {}).get("cited_rule_id") == "R11" and is_ambiguous_presentation(prompt, history):
            log.info("Gemini produced initial R11 for ambiguous complaint — activating follow-up protocol")
            parsed = generate_followup_response(prompt, history)
        else:
            parsed["requires_followup"] = False

    # Append this exchange to the running history
    updated_history = list(history) + [
        {"role": "user",  "parts": [prompt]},
        {"role": "model", "parts": [json.dumps(parsed)]},
    ]
    return parsed, updated_history


def _escalation_fallback(reason: str) -> dict:
    """Return a safe R11 triage note when the model output cannot be trusted."""
    return {
        "status": "triage_note",
        "requires_followup": False,
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
    Initial triage assessment or continuation.
    Accepts: { description, complaint, message, age, history, session_id }
    Returns: { history, status, requires_followup, follow_up_questions?, triage_note? }
    """
    body        = request.get_json(force=True) or {}
    description = (body.get("description") or body.get("complaint") or body.get("message") or "").strip()
    age         = (body.get("age") or "").strip() or None
    history     = body.get("history") or []

    if not description:
        return jsonify({"error": "Patient description is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        log.info("Gemini not configured (%s) — using local deterministic engine", exc)
        client = None

    if history:
        prompt = description
    else:
        age_str = f"\nPatient age: {age}" if age else ""
        prompt = (
            f"Patient intake description:\n\"{description}\"{age_str}\n\n"
            "Analyse this against the triage rules. If the complaint is ambiguous or lacks critical "
            "details (e.g. pain localization, quadrant, onset, vitals), ask targeted follow-up "
            "clarification questions (status: 'follow_up'). If you have sufficient information to safely "
            "route the patient, produce a triage_note."
        )

    result, updated_history = call_gemini(client, prompt, history)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        response["status"] = "triage_note"
        response["requires_followup"] = False
        response["triage_note"] = result["triage_note"]
    elif result.get("status") == "follow_up":
        response["status"] = "follow_up"
        response["requires_followup"] = True
        response["follow_up_questions"] = result.get("follow_up_questions", [])
        response["partial_reasoning"]   = result.get("partial_reasoning", "")
    else:
        response["status"] = "follow_up"
        response["requires_followup"] = True
        response["follow_up_questions"] = [
            "Where is the symptom located right now?",
            "How long have you had this symptom, and was the onset sudden or gradual?",
            "Are you experiencing any other symptoms such as fever, nausea, or dizziness?",
        ]
        response["partial_reasoning"] = "Clarification required to evaluate clinical routing criteria."

    return jsonify(response)


@app.route("/api/followup", methods=["POST"])
def api_followup():
    """
    Continue a triage conversation.
    Accepts: { session_id, message, history }
    Returns: { history, status, requires_followup, assistant_message, triage_note?, follow_up_questions? }
    """
    body    = request.get_json(force=True) or {}
    message = (body.get("message") or body.get("complaint") or body.get("description") or "").strip()
    history = body.get("history") or []

    if not message:
        return jsonify({"error": "Message is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        log.info("Gemini not configured (%s) — using local deterministic engine", exc)
        client = None

    result, updated_history = call_gemini(client, message, history)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        note = result["triage_note"]
        response["status"] = "triage_note"
        response["requires_followup"] = False
        response["triage_note"] = note
        response["assistant_message"] = (
            f"Based on the clarification provided, I have generated a grounded triage dossier "
            f"(Rule {note.get('cited_rule_id', 'N/A')}: {note.get('rule_label', '')}). "
            "Please review the Triage Note panel below."
        )
    elif result.get("status") == "follow_up":
        questions = result.get("follow_up_questions", [])
        response["status"] = "follow_up"
        response["requires_followup"] = True
        response["follow_up_questions"] = questions
        response["partial_reasoning"]   = result.get("partial_reasoning", "")
        lines = ["Thank you. I need a few more details:\n"]
        lines += [f"{i+1}. {q}" for i, q in enumerate(questions)]
        if result.get("partial_reasoning"):
            lines += [f"\n_{result['partial_reasoning']}_"]
        response["assistant_message"] = "\n".join(lines)
    else:
        response["status"] = "follow_up"
        response["requires_followup"] = False
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
        log.info("Gemini not configured (%s) — using local deterministic engine", exc)
        client = None

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
        response["status"] = "triage_note"
        response["requires_followup"] = False
        response["triage_note"] = result["triage_note"]
    else:
        response["status"] = "triage_note"
        response["requires_followup"] = False
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
    Accepts: { complaint, description, message, age, history, session_id }
    Delegates to the same logic as /api/assess.
    """
    body        = request.get_json(force=True) or {}
    description = (body.get("complaint") or body.get("description") or body.get("message") or "").strip()
    age         = (body.get("age") or "").strip() or None
    history     = body.get("history") or []

    if not description:
        return jsonify({"error": "complaint is required."}), 400

    try:
        client = get_gemini_client()
    except EnvironmentError as exc:
        log.info("Gemini not configured (%s) — using local deterministic engine", exc)
        client = None

    if history:
        prompt = description
    else:
        age_str = f"\nPatient age: {age}" if age else ""
        prompt = (
            f"Patient intake description:\n\"{description}\"{age_str}\n\n"
            "Analyse this against the triage rules. If the complaint is ambiguous or lacks critical "
            "details (e.g. pain localization, quadrant, onset, vitals), ask targeted follow-up "
            "clarification questions (status: 'follow_up'). If you have sufficient information to safely "
            "route the patient, produce a triage_note."
        )

    result, updated_history = call_gemini(client, prompt, history)
    response: dict = {"history": updated_history}

    if result.get("status") == "triage_note":
        response["status"] = "triage_note"
        response["requires_followup"] = False
        response["triage_note"] = result["triage_note"]
    elif result.get("status") == "follow_up":
        response["status"] = "follow_up"
        response["requires_followup"] = True
        response["follow_up_questions"] = result.get("follow_up_questions", [])
        response["partial_reasoning"]   = result.get("partial_reasoning", "")
    else:
        response["status"] = "follow_up"
        response["requires_followup"] = True
        response["follow_up_questions"] = [
            "Where is the symptom located right now?",
            "How long have you had this symptom, and was the onset sudden or gradual?",
            "Are you experiencing any other symptoms such as fever, nausea, or dizziness?",
        ]
        response["partial_reasoning"] = "Clarification required to evaluate clinical routing criteria."

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
