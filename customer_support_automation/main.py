import json
import re
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from customer_support import crew, inquiry_resolution
from linkedin.linkedin_crew import crew as linked_in_crew
from eventPlanner.planner_crew import event_management_crew, VenueDetails
from outreach.outreach_crew import outreach_crew

# —───────────── Logging Setup ─────────────—
logger = logging.getLogger("uvicorn.error")


# —───────────── Pydantic Models ─────────────—
class InquiryRequest(BaseModel):
    customer: str = Field(..., example="Maple Grove Dental")
    inquiry: str = Field(..., example="I'd love to learn more about your pricing tiers.")

class InquiryResponse(BaseModel):
    response: str


class GeneratePostRequest(BaseModel):
    topic: str = Field(..., example="AI-driven threat detection")

class GeneratePostResponse(BaseModel):
    response: str


class EventRequest(BaseModel):
    event_topic: str
    event_description: str
    event_city: str
    tentative_date: str  # e.g. "2025-06-15"
    expected_participants: int
    budget: float
    venue_type: str

class MarketingBundle(BaseModel):
    venueDetails: VenueDetails
    marketingReport: str


class OutreachEmailRequest(BaseModel):
    lead_name: str
    industry: str
    recipient_name: str
    recipient_position: str
    recent_event: str
    core_feature: Optional[str] = Field(
        None, description="Defaults to OneAPI data integration if blank"
    )

class OutreachEmailResponse(BaseModel):
    email: str


# —───────────── FastAPI App & CORS ─────────────—
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello World"}

# —───────────── Helpers ─────────────—
async def kickoff_crew(crew, payload: dict) -> str:
    try:
        return await run_in_threadpool(crew.kickoff, payload)
    except Exception as e:
        logger.error("Crew kickoff failed", exc_info=e)
        raise HTTPException(status_code=502, detail="Upstream AI service error")


# —───────────── Routes ─────────────—
@app.post("/chat", response_model=InquiryResponse)
async def chat(req: InquiryRequest):
    # Render the prompt
    template = inquiry_resolution.description
    try:
        rendered = template.format(**req.dict())
    except KeyError as e:
        msg = f"Missing template key: {e}"
        logger.error(msg)
        raise HTTPException(status_code=400, detail=msg)

    logger.debug("Rendered chat prompt: %s", rendered)
    ai_response = await kickoff_crew(crew, req.dict())
    return {"response": ai_response}


@app.post("/generate-post", response_model=GeneratePostResponse)
async def generate_post(req: GeneratePostRequest):
    ai_response = await kickoff_crew(linked_in_crew, {"topic": req.topic})
    return {"response": ai_response}


@app.post("/plan-event", response_model=MarketingBundle)
async def plan_event(req: EventRequest):
    # 1) Kick off the crew
    raw = await kickoff_crew(event_management_crew, req.dict())

    # 2) Extract JSON from first “{” to last “}”
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        logger.error("No JSON object in plan-event output:\n%s", raw)
        raise HTTPException(500, "Invalid agent response format")

    body = raw[start : end + 1]

    # 3) Strip ```json fences if the agent used them
    if body.startswith("```"):
        lines = body.splitlines()[1:-1]
        body = "\n".join(lines)

    # 4) Escape control chars inside the marketingReport value
    def _escape_report(match):
        prefix, report, suffix = match.group(1), match.group(2), match.group(3)
        # escape backslashes and quotes
        report = report.replace("\\", "\\\\").replace('"', '\\"')
        # escape newlines
        report = report.replace("\r\n", "\\n").replace("\n", "\\n")
        return f'{prefix}{report}{suffix}'

    body = re.sub(
        r'("marketingReport"\s*:\s*")([\s\S]*?)(")',
        _escape_report,
        body,
        flags=re.MULTILINE,
    )

    # 5) Parse JSON
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON:\n%s\nError: %s", body, e)
        raise HTTPException(500, "Error parsing AI output")

    # 6) Wrap the Markdown in fences (so frontend can strip them cleanly)
    report = data.get("marketingReport", "")
    data["marketingReport"] = f"```\n{report}\n```"

    # 7) Return validated Pydantic model
    return MarketingBundle(**data)


@app.post("/outreach-email", response_model=OutreachEmailResponse)
async def outreach_email(req: OutreachEmailRequest):
    email = await kickoff_crew(outreach_crew, req.dict())

    return {"email": email}
