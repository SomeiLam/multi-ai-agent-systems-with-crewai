from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SeleniumScrapingTool
import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# ── Agent Definitions ───────────────────────────────────────────────
# Primary support agent: answers developer questions using official docs
support_agent = Agent(
    role="Sikka API Support Engineer",
    goal=(
        "Provide clear, accurate, and actionable guidance on using the "
        "Sikka OneAPI and related developer endpoints, "
        "and never invent information."
    ),
    backstory=(
        "You are the lead developer support engineer at Sikka.ai. "
        "Developers (like {person} at {customer}) come to you with questions "
        "about integrating and using Sikka APIs (https://www.sikka.ai/, "
        "https://www.sikka.ai/oneapi, https://apidocs.sikkasoft.com/). "
        "Your mission is to deliver complete, no-assumptions answers. "
        "If you cannot find an answer in the docs or tools, respond honestly: "
        "'I’m sorry, I couldn’t find that information in the Sikka documentation.'"
    ),
    allow_delegation=False,
    verbose=True
)

# QA agent: verifies technical correctness and completeness
support_qa_agent = Agent(
    role="Sikka Support QA Specialist",
    goal=(
        "Ensure every support response is technically correct, comprehensive, "
        "and follows best practices for developer docs; do not add false info."
    ),
    backstory=(
        "You are the Quality Assurance lead for Sikka developer support. "
        "After the Support Engineer drafts an answer for {customer}, "
        "you review to confirm all API endpoints, parameters, error codes, "
        "and sample requests are accurate and complete, referring to docs as needed. "
        "If any information is missing or unclear, instruct to return: "
        "'I’m sorry, I couldn’t find that information in the Sikka documentation.'"
    ),
    allow_delegation=False,
    verbose=True
)

# ── Tool Definitions ───────────────────────────────────────────────
# Scrapers for the three Sikka documentation sites
docs_tool    = ScrapeWebsiteTool(website_url="https://www.sikka.ai/")
oneapi_tool  = ScrapeWebsiteTool(website_url="https://www.sikka.ai/oneapi")
# Headless browser to render JS-driven API docs
selenium_tool = SeleniumScrapingTool(
    website_url="https://apidocs.sikkasoft.com/"
)

tools = [docs_tool, oneapi_tool, selenium_tool]

# ── Task Definitions ───────────────────────────────────────────────

# First task: draft the support answer
inquiry_resolution = Task(
    description=(
        "{customer}'s developer {person} asks:\n"
        "{inquiry}\n\n"
        "Using the Sikka API docs and any example code, provide a step-by-step "
        "integration guide or troubleshooting steps. Include code snippets, "
        "endpoint URLs, query parameters, and sample responses. Do not assume prior knowledge—"
        "explain every step clearly. If info is unavailable, say so."
    ),
    expected_output=(
        "A complete, friendly support answer that:\n"
        "- References the exact Sikka API endpoints used\n"
        "- Shows example request and response payloads\n"
        "- Explains authentication or configuration needed\n"
        "- Points to relevant docs URLs for deeper reading"
    ),
    tools=tools,
    agent=support_agent,
)

# Second task: QA review of the draft answer
quality_assurance_review = Task(
    description=(
        "Review the draft support answer for {customer}'s inquiry. "
        "Check that every endpoint, parameter, and example is correct, "
        "and that the explanation is thorough and easy to follow. "
        "Verify links to the Sikka docs and adjust any inaccuracies. "
        "If something cannot be confirmed, add an honest note."
    ),
    expected_output=(
        "A polished, error-free final response ready to send, "
        "with confirmed code samples and doc references. "
        "Maintain a professional yet approachable tone."
    ),
    tools=tools,
    agent=support_qa_agent,
)

# ── Crew Assembly & Execution ──────────────────────────────────────
crew = Crew(
    agents=[support_agent, support_qa_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=1,
    memory=True
)

# Input parameters for a sample inquiry
inputs = {
    "customer": "AcmeHealth",
    "person": "Jane Doe",
    "inquiry": (
        "What HTTP method, URL path and headers do I need to call the "
        "`/authorized_practices` endpoint? "
        "Please include a Python `requests` snippet showing how to authenticate "
        "and fetch the list of practices, and show a sample JSON response schema."
    )
}

# Fetching patient demographics
inputs1 = {
    "customer": "MediClinic",
    "person": "Dr. Smith",
    "inquiry": (
        "How do I retrieve patient demographic details using the `/v2/patient/{patient_id}/demographics` "
        "endpoint? Please provide the full URL pattern, required headers, and a Python `requests` example, "
        "including how to handle a 404 if the patient doesn’t exist."
    )
}

# Posting a new appointment
inputs2 = {
    "customer": "HealthPlus",
    "person": "Emily Chen",
    "inquiry": (
        "What JSON payload do I need to send to the `/v2/appointments` POST endpoint to schedule a new appointment? "
        "Please include a Python snippet showing the request body format, authentication headers, and sample success "
        "and error responses."
    )
}

# Error handling for rate limits
inputs3 = {
    "customer": "WellCare",
    "person": "Carlos Rivera",
    "inquiry": (
        "What status code and response body does the API return when I exceed the rate limit? "
        "How should I implement an exponential backoff retry in Python `requests` to handle it?"
    )
}

# Filtering practices by specialty
inputs4 = {
    "customer": "PrimeHealth",
    "person": "Olivia Patel",
    "inquiry": (
        "Can I filter the `/v2/authorized_practices` endpoint by specialty or location? "
        "If supported, what query parameters should I include, and can you show a sample URL and Python snippet?"
    )
}

if __name__ == "__main__":
    # Kick off the multi-agent workflow and print the result
    result = crew.kickoff(inputs=inputs)
    print(result)
