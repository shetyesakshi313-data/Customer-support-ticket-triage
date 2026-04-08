"""
FastAPI server — exposes OpenEnv HTTP endpoints:
  GET  /
  GET  /health
  POST /reset
  POST /step
  GET  /state
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from server.my_env_v4 import (
    MyEnvV4Env,
    MyEnvV4Action,
    MyEnvV4Observation,
    MyEnvV4State,
)

TASK = os.getenv("MY_ENV_V4_TASK", "easy")

app = FastAPI(
    title="Support Ticket Triage — OpenEnv",
    description="Real-world customer support triage environment for agentic RL.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create environment
_env = MyEnvV4Env(task=TASK)


# ROOT ENDPOINT (Homepage)
@app.get("/")
async def root():
    return {
        "message": "Support Ticket Triage OpenEnv is running",
        "available_endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/docs"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "task": TASK}


@app.post("/reset", response_model=MyEnvV4Observation)
async def reset():
    obs = await _env.reset()
    return obs


@app.post("/step", response_model=dict)
async def step(action: MyEnvV4Action):
    obs, reward, done, info = await _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=MyEnvV4State)
async def state():
    return await _env.state()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False
    )