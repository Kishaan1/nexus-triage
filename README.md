TRACK_ID=PS01
# 🏥 NexusTriage-OS
### Autonomous Grounded Patient Intake & Clinical Routing Engine

[![Python 3](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)](https://nexus-triage-1.onrender.com)
[![Render](https://img.shields.io/badge/Render-Live-46E3B7?logo=render&logoColor=white)](https://nexus-triage-1.onrender.com)
[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo%20Video-FF0000?logo=youtube&logoColor=white)](https://youtu.be/BOXTUyqze84)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/Kishaan1/nexus-triage)

🌐 **Render Live Link:** [https://nexus-triage-1.onrender.com](https://nexus-triage-1.onrender.com)  
🎥 **Demo Video:** [https://youtu.be/BOXTUyqze84](https://youtu.be/BOXTUyqze84)  
📂 **Repository:** [https://github.com/Kishaan1/nexus-triage](https://github.com/Kishaan1/nexus-triage)  

---

## 📌 Project Overview

- 🏷️ **Project Name:** NexusTriage-OS
- 🎯 **Track ID:** PS01 (Healthcare - Patient Intake Triage Assistant)
- 📂 **Repository:** [https://github.com/Kishaan1/nexus-triage](https://github.com/Kishaan1/nexus-triage)
- 🎥 **Demo Video:** [https://youtu.be/BOXTUyqze84](https://youtu.be/BOXTUyqze84)
- 🌐 **Live Deployment:** [https://nexus-triage-1.onrender.com](https://nexus-triage-1.onrender.com)
- 🛠️ **Tech Stack:** Python, Flask, Google Gemini API, HTML/CSS/JavaScript

---

## 🏗️ Project Context & System Architecture

NexusTriage-OS is a clinical intake triage and routing engine designed for Track PS01. It takes unstructured patient complaints and deterministically maps them to urgency tiers (Immediate Emergency, Urgent, Semi-Urgent, Routine) and destination hospital departments without providing medical diagnoses or drug prescriptions.

### ⚙️ Core Mechanisms

1. 🎯 **Grounded Determinism:** Utilizes `data/triage_rules.json` to enforce deterministic rule citations (such as Rule R05 for acute chest pain/emergency escalation and Rule R11 for clinical ambiguity).
2. 🛡️ **Epistemic Guardrails:** Separates subjective patient-reported symptoms from established clinical observations and unverified assumptions.
3. 💬 **Interactive Clarification:** When critical clinical parameters are missing, the engine triggers an interactive follow-up clarification protocol rather than making speculative assumptions.
4. 🤖 **LLM Role:** Powered by Google Gemini via the official SDK strictly for natural language entity parsing and conversational follow-up questions, leaving routing decisions grounded in deterministic clinical rules.

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
