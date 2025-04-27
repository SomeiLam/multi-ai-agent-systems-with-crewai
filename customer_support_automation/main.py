from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from crew_setup import crew
from crew_setup import crew, inquiry_resolution

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class InquiryRequest(BaseModel):
    customer: str
    person: str
    inquiry: str

@app.post("/chat")
async def chat(req: InquiryRequest):
    """
    Kick off the multi-agent workflow with the given inputs
    and return the assembled result.
    """

    payload = {
        "customer": req.customer,
        "person": req.person,
        "inquiry": req.inquiry,
    }

    # 3) Manually render the prompt text from the Task template
    raw_template = inquiry_resolution.description
    try:
        rendered = raw_template.format(**payload)
    except KeyError as e:
        rendered = f"❌ Failed to render template, missing key: {e}"
    print("\n\n>>> RENDERED PROMPT TO LLM:\n")
    print(rendered)
    print("\n<<< END RENDERED PROMPT >>>\n\n")

    # 4) Finally call kickoff
    result = await run_in_threadpool(crew.kickoff, payload)
    return {"response": result}