TRACK_ID=PS01
# NexusTriage-OS
### Autonomous Grounded Patient Intake & Clinical Routing Engine

[![Python 3](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://nexus-triage-1.onrender.com)
[![Render](https://img.shields.io/badge/Render-Live-46E3B7?logo=render&logoColor=white)](https://nexus-triage-1.onrender.com)

🌐 **Render Live Link:** [https://nexus-triage-1.onrender.com](https://nexus-triage-1.onrender.com)

NexusTriage-OS is a clinical intake triage assistant built for **NexusTiQ 24 (Track PS01)**. It translates patient walk-in complaints into structured clinical triage dossiers without providing medical diagnoses.

---

## ⚡ Quickstart (Single-Command Execution)

The entire application runs as a self-contained local web server binding to port `8000`.

### 1. Install Minimal Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variable
Create a `.env` file or export your Gemini API key:
```bash
# Linux / macOS
export GEMINI_API_KEY="your-api-key"

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-api-key"

# Windows (CMD)
set GEMINI_API_KEY=your-api-key
```

### 3. Launch the Server
```bash
python app.py
```
Access the application at `http://localhost:8000` (binding to `0.0.0.0:8000`).

---

## 🛡️ Grounded Deterministic Clinical Routing & Zero-Diagnosis Guardrails

NexusTriage-OS operates strictly as a **Clinical Routing Engine** and never provides medical diagnoses, treatments, or speculative clinical statements:
- **12 Grounded Protocols**: Directly evaluates presentations against `data/triage_rules.json` (Rule IDs `R01`–`R12`).
- **Auditable Citations**: Every recommendation cites exact Rule IDs (e.g., `Per Rule R05`) and auditable reasoning.
- **Sentinel Gateway**: Ambiguous or high-risk cases automatically trigger follow-up questions or Rule `R11` (`ESCALATE`) requiring immediate clinician assessment.
- **Administrative Boundary Handling**: Non-medical requests (billing, appointments, paperwork) route cleanly to Patient Services (`R12`).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the elevated cyber-clinical single-page web interface |
| `POST` | `/api/triage` | Single-turn triage assessment (`{ "complaint": "...", "age": "..." }`) |
| `POST` | `/api/assess` | Multi-turn conversational intake endpoint |
| `POST` | `/api/followup` | Interactive follow-up messaging and question answering |
| `POST` | `/api/finalize` | Force-generate final structured clinical triage note |
| `GET` | `/api/rules` | Inspect loaded local clinical triage rules matrix |
| `GET` | `/api/health` | Health check confirmation (`status: ok`, `track_id: PS01`) |

---

## 📋 Track PS01 Submission Checklist

- [x] Line 1 of `README.md` is strictly `TRACK_ID=PS01`
- [x] Single-command local execution on port `8000` (`python app.py`)
- [x] Exclusively uses official Google Gemini SDK (`google-genai`)
- [x] Zero third-party vector databases, external search engines, or closed SaaS wrappers
- [x] Strictly grounded in local `data/triage_rules.json`
- [x] Minimal production dependencies in `requirements.txt`
