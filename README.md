TRACK_ID=PS01

# NexusTriage-OS

[![Live Demo](https://img.shields.io/badge/Live_Demo-Available-22e8f0?style=for-the-badge&logo=render&logoColor=black)](https://nexus-triage-1.onrender.com)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://html.com/)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://www.w3.org/Style/CSS/)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://javascript.info/)


## Architectural Summary
NexusTriage-OS is a state-of-the-art Patient Intake Triage Assistant built using a lightweight Flask backend and a highly dynamic, zero-dependency HTML5/CSS3/Vanilla JS frontend. The system acts as a "Grounded Clinical Routing Engine." 
It bridges the gap between unstructured, plain-language patient complaints and a deterministic set of clinical triage rules (stored locally in JSON format). It uses Google's `gemini-3.5-flash-lite` model for natural language understanding and structuring, while rigidly enforcing safety through system instructions and grounding it strictly to the available ruleset.

## Quickstart

To run the application locally on your machine, simply define your Gemini API key as an environment variable and launch the single entry point.

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your_key_here
python app.py
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your_key_here"
python app.py
```

**Mac / Linux:**
```bash
export GEMINI_API_KEY=your_key_here
python app.py
```

*The application will boot in under 90 seconds and automatically bind to `http://localhost:8000`.*

## Clinical Grounding & Non-Diagnostic Constraints
NexusTriage-OS is strictly prohibited from diagnosing conditions or recommending treatments. 
The system operates exclusively as a **Routing Tool**. It maps a patient's symptoms to a set of 12 predefined deterministic triage rules. The AI evaluates the input, and if it matches a known rule, it cites the specific `Rule ID` and the exact reasoning based on that rule. If a patient presents a life-threatening or highly ambiguous condition, the system automatically defaults to an `ESCALATE` or `URGENT` tier, explicitly requiring a human clinician to step in.

## Follow-up Protocol for Ambiguous Cases
Patients rarely provide all the required information in their initial complaint. If the intake description lacks sufficient detail to confidently match a triage rule (e.g., missing pain scale, onset duration, or associated symptoms), the assistant will not guess or assume. Instead, it enters a structured **Follow-up Chat Protocol**. It asks targeted clarification questions (e.g., "Is the pain spreading anywhere?" or "On a scale of 1-10, how severe is it?"). The conversation continues until the system gathers enough data to safely trigger a final triage rule.

## Demonstration Test Cases
The frontend includes 4 "Quick-Load Scenarios" to demonstrate the breadth of the triage engine:

1. **🚨 Emergency Chest Pain** (Rule `RULE-CP-01`): Demonstrates immediate detection of high-risk symptoms, instantly routing the patient to the Emergency department with an `IMMEDIATE` urgency tier.
2. **🌡️ Routine Mild Fever** (Rule `RULE-F-01`): Demonstrates a standard, low-risk assessment routing the patient to General Practice.
3. **❓ Ambiguous Abdominal Pain** (Rule `RULE-AB-01` / Escalation): Demonstrates the system's safety nets. Unclear abdominal pain triggers the Follow-up Protocol to gather more data, and if uncertainty persists, routes as `URGENT` or escalates to a human clinician.
4. **🚫 Administrative Null Case** (Rule `RULE-ADMIN-01`): Demonstrates robust boundary handling. When a patient asks for paperwork or billing, the system correctly identifies it as a `NOT_MEDICAL` request and routes them to Administration.

## 🚀 Submission Checklist
- [x] Single-command startup on port 8000
- [x] Gemini API non-diagnostic clinical routing
- [x] Follow-up protocol for ambiguous inputs
