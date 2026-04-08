"""
Customer Support Ticket Triage Environment — OpenEnv compliant
Real-world task: An AI agent reads support tickets and must:
  - Classify priority (low/medium/high/critical)
  - Route to the correct department
  - Generate a short resolution plan

Tasks:
  easy   → Classify priority only
  medium → Classify priority + route to department
  hard   → Classify + route + generate resolution steps
"""

import random
import textwrap
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel


# ──────────────────────────────────────────────
# Data models
# ──────────────────────────────────────────────

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class Department(str, Enum):
    billing = "billing"
    technical = "technical"
    account = "account"
    shipping = "shipping"
    general = "general"


class MyEnvV4Action(BaseModel):
    """Action the agent submits."""
    priority: Optional[Priority] = None          # used in easy+
    department: Optional[Department] = None      # used in medium+
    resolution_steps: Optional[list[str]] = None # used in hard


class MyEnvV4Observation(BaseModel):
    """What the agent sees."""
    ticket_id: str
    ticket_text: str
    task: str          # "easy" | "medium" | "hard"
    step: int
    message: str = ""


class MyEnvV4State(BaseModel):
    episode_id: str
    task: str
    step: int
    done: bool
    cumulative_reward: float


# ──────────────────────────────────────────────
# Ticket corpus  (20 varied tickets)
# ──────────────────────────────────────────────

TICKETS = [
    {
        "text": "My account has been charged twice for the same order #4521. Please refund immediately.",
        "priority": Priority.high,
        "department": Department.billing,
        "resolution": ["Verify duplicate charge in payment system", "Issue refund to original payment method", "Send confirmation email to customer"],
    },
    {
        "text": "I forgot my password and cannot log in. The reset email never arrives.",
        "priority": Priority.medium,
        "department": Department.account,
        "resolution": ["Check spam filters", "Resend password reset to verified email", "Escalate if email delivery fails"],
    },
    {
        "text": "The entire production API is returning 500 errors for all customers since 10 minutes ago.",
        "priority": Priority.critical,
        "department": Department.technical,
        "resolution": ["Page on-call engineer immediately", "Check error logs and recent deployments", "Roll back last deploy if needed", "Post status update"],
    },
    {
        "text": "My package was supposed to arrive 3 days ago but tracking shows no update.",
        "priority": Priority.medium,
        "department": Department.shipping,
        "resolution": ["Check carrier tracking system", "Contact carrier for investigation", "Offer replacement shipment if lost"],
    },
    {
        "text": "How do I export my data as a CSV?",
        "priority": Priority.low,
        "department": Department.general,
        "resolution": ["Point customer to help docs on data export", "Send direct link to CSV export feature"],
    },
    {
        "text": "I was charged for a plan I never signed up for. This is fraud!",
        "priority": Priority.high,
        "department": Department.billing,
        "resolution": ["Verify subscription history", "Refund unauthorized charge", "Check for account compromise", "Notify security team"],
    },
    {
        "text": "App crashes every time I try to upload a file larger than 10MB.",
        "priority": Priority.medium,
        "department": Department.technical,
        "resolution": ["Reproduce the bug", "Check file upload limits in config", "Deploy hotfix or workaround"],
    },
    {
        "text": "Can I change the email address on my account?",
        "priority": Priority.low,
        "department": Department.account,
        "resolution": ["Guide customer to account settings", "Verify new email ownership"],
    },
    {
        "text": "URGENT: All our team members are locked out of the system. We have a board meeting in 1 hour.",
        "priority": Priority.critical,
        "department": Department.account,
        "resolution": ["Escalate to senior support immediately", "Manually unlock accounts", "Investigate root cause after resolution"],
    },
    {
        "text": "I received the wrong item in my order. I ordered a blue shirt but got a red one.",
        "priority": Priority.medium,
        "department": Department.shipping,
        "resolution": ["Apologize and confirm wrong item", "Arrange return pickup", "Ship correct item expedited"],
    },
    {
        "text": "What are your business hours?",
        "priority": Priority.low,
        "department": Department.general,
        "resolution": ["Provide business hours information"],
    },
    {
        "text": "Payment gateway is down. None of our customers can checkout. We are losing revenue.",
        "priority": Priority.critical,
        "department": Department.technical,
        "resolution": ["Page payments team immediately", "Switch to backup payment processor", "Post incident status"],
    },
    {
        "text": "I was overcharged by $5.00 on my last invoice.",
        "priority": Priority.medium,
        "department": Department.billing,
        "resolution": ["Review invoice line items", "Issue $5 credit or refund"],
    },
    {
        "text": "My account shows a different username than what I registered with.",
        "priority": Priority.low,
        "department": Department.account,
        "resolution": ["Verify registration record", "Correct username in database"],
    },
    {
        "text": "Shipment tracking link is broken — returns a 404 error.",
        "priority": Priority.medium,
        "department": Department.shipping,
        "resolution": ["Fix broken tracking URL", "Provide direct tracking number to customer"],
    },
    {
        "text": "I'd like to cancel my subscription.",
        "priority": Priority.low,
        "department": Department.billing,
        "resolution": ["Confirm cancellation intent", "Process cancellation", "Send confirmation email"],
    },
    {
        "text": "Two-factor authentication code is not being delivered to my phone.",
        "priority": Priority.high,
        "department": Department.account,
        "resolution": ["Check SMS provider status", "Offer backup codes or email 2FA", "Escalate if systemic"],
    },
    {
        "text": "Data sync between mobile and desktop is completely broken for all users.",
        "priority": Priority.critical,
        "department": Department.technical,
        "resolution": ["Declare incident", "Check sync service health", "Roll back or patch sync service"],
    },
    {
        "text": "I need an invoice for my company for tax purposes.",
        "priority": Priority.low,
        "department": Department.billing,
        "resolution": ["Generate and email official invoice with company details"],
    },
    {
        "text": "The mobile app is draining battery abnormally fast after the latest update.",
        "priority": Priority.medium,
        "department": Department.technical,
        "resolution": ["Log bug report", "Profile battery usage in latest build", "Release patch update"],
    },
]


# ──────────────────────────────────────────────
# Environment
# ──────────────────────────────────────────────

class MyEnvV4Env:
    """
    OpenEnv-compliant environment for Customer Support Ticket Triage.
    Supports three tasks: easy / medium / hard.
    """

    def __init__(self, task: str = "easy", image_name: Optional[str] = None):
        assert task in ("easy", "medium", "hard"), f"Unknown task: {task}"
        self.task = task
        self._ticket: Optional[dict] = None
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._episode_id = ""

    async def reset(self) -> MyEnvV4Observation:
        self._ticket = random.choice(TICKETS)
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._episode_id = f"ep_{random.randint(10000, 99999)}"
        return self._make_obs("New ticket assigned. Analyze and respond.")

    async def step(self, action: MyEnvV4Action) -> tuple[MyEnvV4Observation, float, bool, dict]:
        self._step += 1
        reward = 0.0
        error = None

        try:
            if self.task == "easy":
                reward, error = self._grade_easy(action)
                self._done = True

            elif self.task == "medium":
                reward, error = self._grade_medium(action)
                self._done = True

            elif self.task == "hard":
                reward, error = self._grade_hard(action)
                self._done = True

        except Exception as exc:
            error = str(exc)
            self._done = True

        self._cumulative_reward += reward
        obs = self._make_obs(error or "Step complete.")
        return obs, reward, self._done, {"last_action_error": error}

    async def state(self) -> MyEnvV4State:
        return MyEnvV4State(
            episode_id=self._episode_id,
            task=self.task,
            step=self._step,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
        )

    async def close(self) -> None:
        pass

    # ── graders ──────────────────────────────

    def _grade_easy(self, action: MyEnvV4Action) -> tuple[float, Optional[str]]:
        """Grade priority classification only."""
        if action.priority is None:
            return 0.0, "priority field is required for easy task"
        correct = self._ticket["priority"]
        if action.priority == correct:
            return 1.0, None
        # partial credit for adjacent priority
        levels = [Priority.low, Priority.medium, Priority.high, Priority.critical]
        diff = abs(levels.index(action.priority) - levels.index(correct))
        partial = max(0.0, 1.0 - diff * 0.35)
        return round(partial, 2), f"Expected {correct}, got {action.priority}"

    def _grade_medium(self, action: MyEnvV4Action) -> tuple[float, Optional[str]]:
        """Grade priority + department routing."""
        priority_score, err = self._grade_easy(action)
        dept_score = 0.0
        if action.department is None:
            return priority_score * 0.5, "department field is required for medium task"
        if action.department == self._ticket["department"]:
            dept_score = 1.0
        total = 0.5 * priority_score + 0.5 * dept_score
        expected_dept = self._ticket["department"]

        return round(total, 2), None if total == 1.0 else (
        f"priority={'correct' if priority_score == 1 else 'wrong'}, "
        f"dept={'correct' if dept_score == 1 else 'expected ' + str(expected_dept)}"
        )

    def _grade_hard(self, action: MyEnvV4Action) -> tuple[float, Optional[str]]:
        """Grade priority + department + resolution steps."""
        medium_score, _ = self._grade_medium(action)
        resolution_score = 0.0
        if not action.resolution_steps:
            return medium_score * 0.5, "resolution_steps required for hard task"

        expected = self._ticket["resolution"]
        # Score: how many expected keywords appear across submitted steps
        submitted_text = " ".join(action.resolution_steps).lower()
        keyword_hits = sum(
            1 for step in expected
            if any(word in submitted_text for word in step.lower().split() if len(word) > 4)
        )
        resolution_score = min(1.0, keyword_hits / max(len(expected), 1))

        # Length penalty: too short is bad
        if len(action.resolution_steps) < 2:
            resolution_score *= 0.5

        total = 0.4 * medium_score + 0.6 * resolution_score
        return round(total, 2), None

    # ── helpers ──────────────────────────────

    def _make_obs(self, message: str) -> MyEnvV4Observation:
        return MyEnvV4Observation(
            ticket_id=self._episode_id,
            ticket_text=self._ticket["text"] if self._ticket else "",
            task=self.task,
            step=self._step,
            message=message,
        )
