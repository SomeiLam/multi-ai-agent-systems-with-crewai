from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool, SeleniumScrapingTool
import os
from utils import get_openai_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

# ── Agent Definitions ───────────────────────────────────────────────

# Business support specialist: explains Sikka's offerings using marketing copy
support_agent = Agent(
    role="Sikka Business Support Specialist",
    goal=(
        "Provide concise, accurate explanations of Sikka.ai's services, features, "
        "and OneAPI value proposition as described on the website, without fabricating details."
    ),
    backstory=(
        "You are the lead business support specialist at Sikka.ai. "
        "Customers (like {customer}) ask questions about Sikka's offerings, pricing, and use cases "
        "based solely on the content at https://www.sikka.ai/ and https://www.sikka.ai/oneapi. "
        "Use only the marketing copy and high-level descriptions from those pages. "
        "If the information is not found, respond: "
        "‘I’m sorry, I couldn’t find that information on the Sikka.ai pages.’"
    ),
    allow_delegation=False,
    verbose=True
)

# QA specialist: verifies all business statements match the website content
support_qa_agent = Agent(
    role="Sikka Business QA Specialist",
    goal=(
        "Ensure each business support response strictly reflects the marketing and product descriptions "
        "from the Sikka.ai landing and OneAPI pages."
    ),
    backstory=(
        "You are the QA lead for business support at Sikka.ai. "
        "After the Business Support Specialist drafts an answer for {customer}, "
        "you review to confirm all feature descriptions, benefits, and claims exactly match "
        "the content on https://www.sikka.ai/ and https://www.sikka.ai/oneapi. "
        "If any detail cannot be verified, instruct to reply: "
        "‘I’m sorry, I couldn’t find that information on the Sikka.ai pages.’"
    ),
    allow_delegation=False,
    verbose=True
)

# ── Tool Definitions ───────────────────────────────────────────────
# Only scrape the official landing and OneAPI pages
docs_tool   = ScrapeWebsiteTool(website_url="https://www.sikka.ai/")
about_tool  = ScrapeWebsiteTool(website_url="https://www.sikka.ai/about-us")
oneapi_tool = ScrapeWebsiteTool(website_url="https://www.sikka.ai/oneapi")

tools = [docs_tool, about_tool, oneapi_tool]

# ── Task Definitions ───────────────────────────────────────────────

# Task 1: draft a business-focused answer
inquiry_resolution = Task(
    description=(
        "{customer} asks:\n"
        "{inquiry}\n\n"
        "Using only the marketing and product descriptions from https://www.sikka.ai/ "
        "and https://www.sikka.ai/oneapi, provide a customer-friendly explanation of what Sikka offers. "
        "Highlight features, benefits, and potential use cases. "
        "If details are missing, state that the information is not available."
    ),
    expected_output=(
        "A clear, engaging business answer that:\n"
        "- Copies URLs, headings, and feature names exactly from the specified pages\n"
        "- Describes services, pricing models, or use cases as presented\n"
        "- Maintains a warm, professional tone"
    ),
    tools=tools,
    agent=support_agent,
)

# Task 2: QA review of the business answer
quality_assurance_review = Task(
    description=(
        "Review the business support answer for {customer}'s inquiry. "
        "Verify that every statement exactly matches the marketing copy and product details "
        "on https://www.sikka.ai/ and https://www.sikka.ai/oneapi. "
        "Remove any unsupported claims or added details. "
        "If something cannot be confirmed, note it was not found."
    ),
    expected_output=(
        "A final, polished business response strictly based on the landing and OneAPI page content."
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

# Sample inquiry input
inputs = {
    "customer": "AcmeHealth",
    "person": "Jane Doe",
    "inquiry": (
        "What services does Sikka offer for healthcare providers? "
    )
}

if __name__ == "__main__":
    # Kick off the multi-agent workflow and print the result
    result = crew.kickoff(inputs=inputs)
    print(result)

