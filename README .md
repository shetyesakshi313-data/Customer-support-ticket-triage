---
title: Support Triage Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---
# 🎫 Customer Support Ticket Triage — OpenEnv Environment

A real-world **agentic RL environment** where an AI agent reads customer support
tickets and must classify, route, and resolve them — built for the
**Meta × PyTorch × Scaler OpenEnv Hackathon (Round 1)**.

---

## 🌍 Real-World Task

Support ticket triage is a genuine problem every company faces. An agent must:
1. Determine ticket **priority** (low → critical)
2. Route to the correct **department** (billing, technical, account, shipping, general)
3. Generate **resolution steps** appropriate to the ticket

---

## 📊 Tasks

| Task   | Description                              | Difficulty | Score |
|--------|------------------------------------------|------------|-------|
| easy   | Classify priority only                   | Easy       | 0–1.0 |
| medium | Classify priority + route to department  | Medium     | 0–1.0 |
| hard   | Priority + department + resolution steps | Hard       | 0–1.0 |

---

## 🔁 Action / Observation Spaces

**Action** (`POST /step`):
```json
{
  "priority": "low|medium|high|critical",
  "department": "billing|technical|account|shipping|general",
  "resolution_steps": ["step 1", "step 2"]
}
```

**Observation** (`POST /reset` response):
```json
{
  "ticket_id": "ep_12345",
  "ticket_text": "My account was charged twice...",
  "task": "easy",
  "step": 0,
  "message": "New ticket assigned."
}
```

---

## 🏆 Reward Function

- **easy**: 1.0 for exact priority match; partial credit (0.65 / 0.30) for 1/2 levels off
- **medium**: `0.5 × priority_score + 0.5 × department_score`
- **hard**: `0.4 × medium_score + 0.6 × resolution_keyword_coverage`

All scores in `[0.0, 1.0]`.

---

## 🚀 Setup & Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
MY_ENV_V4_TASK=easy uvicorn app:app --host 0.0.0.0 --port 7860

# 3. Test it
curl -X POST http://localhost:7860/reset
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"priority": "high"}'

# 4. Run inference (set your HF_TOKEN first)
export HF_TOKEN=hf_your_token_here
export MY_ENV_V4_TASK=easy
python inference.py
```

---

## 🐳 Docker

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 -e MY_ENV_V4_TASK=easy support-triage-env
```

---

## 📡 API Endpoints

| Method | Path     | Description                  |
|--------|----------|------------------------------|
| GET    | /health  | Health check                 |
| POST   | /reset   | Start new episode            |
| POST   | /step    | Submit action, get reward    |
| GET    | /state   | Current episode state        |

---

## 📁 File Structure

```
.
├── my_env_v4.py      # Core environment logic + graders
├── app.py            # FastAPI HTTP server
├── inference.py      # Mandatory inference script
├── openenv.yaml      # OpenEnv spec declaration
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
