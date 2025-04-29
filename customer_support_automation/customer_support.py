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
        "based solely on the content at https://www.sikka.ai/, https://www.sikka.ai/about-us, "
        "https://www.sikka.ai/oneapi, and https://www.sikka.ai/sikka-prime. "
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
        "from the Sikka.ai landing, About-Us, OneAPI, and Sikka-Prime pages."
    ),
    backstory=(
        "You are the QA lead for business support at Sikka.ai. "
        "After the Business Support Specialist drafts an answer for {customer}, "
        "you review to confirm all feature descriptions, benefits, and claims exactly match "
        "the content on https://www.sikka.ai/, https://www.sikka.ai/about-us, "
        "https://www.sikka.ai/oneapi, and https://www.sikka.ai/sikka-prime. "
        "If any detail cannot be verified, instruct to reply: "
        "‘I’m sorry, I couldn’t find that information on the Sikka.ai pages.’"
    ),
    allow_delegation=False,
    verbose=True
)

# ── Tool Definitions ───────────────────────────────────────────────
# 1) Root homepage only
landing_tool = ScrapeWebsiteTool(
    website_url="https://www.sikka.ai",
    path_regex=r"^/$",        # only “/”
    crawl=False
)

# 2) /about-us only
about_tool = ScrapeWebsiteTool(
    website_url="https://www.sikka.ai",
    path_regex=r"^/about-us$",
    crawl=False
)

# 3) /oneapi only
oneapi_tool = ScrapeWebsiteTool(
    website_url="https://www.sikka.ai",
    path_regex=r"^/oneapi$",
    crawl=False
)
prime_tool = ScrapeWebsiteTool(
    website_url="https://www.sikka.ai",
    path_regex=r"^/sikka-prime$",
    crawl=False
)


# Order matters: more specific tools first
tools = [prime_tool, oneapi_tool, about_tool, landing_tool]

# ── Task Definitions ───────────────────────────────────────────────

# Task 1: draft a business-focused answer without fences and with real new-lines
inquiry_resolution = Task(
    description=(
        """You are a Sikka.ai live support chat agent.

        Answer every question using *only* the exact marketing copy and feature names found on these pages:
          • https://www.sikka.ai/
          • https://www.sikka.ai/about-us
          • https://www.sikka.ai/oneapi
          • https://www.sikka.ai/sikka-prime

        INPUT:
        - customer: {customer}
        - inquiry:  {inquiry}

        === HOW TO ANSWER ===
        1. **Understand Intent:** Identify the core concept(s) in the user’s inquiry—even if their wording doesn’t exactly match the site.
        2. **Find Source Text:** Search _all_ scraped content for any sentence(s) or phrase(s) that define or describe those concepts.  
          - You may match synonyms or rephrased versions as long as the meaning aligns.
        3. **If you find ≥1 relevant passage:**
          a. **No greetings or pleasantries:** Start with a plural-voice sentence that directly answers the question.
          b. **Intro:** Write one first-person plural sentence that directly answers the question.  
          c. **Key points:** One blank line, then `**Key points:**`, then up to 5 bullet points (`- `) quoting each matched passage verbatim.  
          d. **Source:** One blank line, then `Source: <URL>` showing the exact page you pulled the passages from.  
          e. One blank line, then exactly:
                Do you have any other questions I can help with?

        4. **If you find nothing relevant:**  
          Reply _only_ with:
        "I’m sorry, I couldn’t find that information on the Sikka.ai pages. However, here are some other key features that may help:

        Key points:

        - <Bullet 1: another feature verbatim>
        - <Bullet 2: another feature verbatim> [...]

        Source: <URL>

        Do you have any other questions I can help with?"
        """
    ),
    expected_output=(
        "A raw Markdown response with:\n"
        "- One plural-voice intro sentence\n"
        "- One sentence on why it matters\n"
        "- **Key points:** + 3–5 verbatim bullets\n"
        "- A **Use case example** sentence\n"
        "- A **Source:** line\n"
        "- The closing question\n"
        "- Or just the apology if info is missing"
    ),
    tools=tools,
    agent=support_agent,
)


# Task 2: QA review of the Markdown formatting and content
quality_assurance_review = Task(
    description=(
        """You are the QA lead for Sikka.ai live support chat.  
Your job is to take the **full** Markdown reply from the Business Support Specialist and make sure it:

1. Follows the template (intro, **Key points:**, bullets, Source line, closing question).  
2. Uses first-person plural voice.  
3. Quotes features **verbatim** from one of the four pages.

**If you find formatting issues** (missing heading, bullets, blank lines, etc.), correct them in place.  
**Remove** any bullet point that cannot be verified on the specified pages.  
**Do not** ever replace the entire reply with an apology—that’s only for the support agent itself.  
Always return the **complete** corrected response as raw Markdown."""
    ),
    expected_output=(
        "A raw Markdown string containing:\n"
        "- The original or corrected intro\n"
        "- A **Key points:** section with only verified bullets\n"
        "- A correct Source line\n"
        "- The closing question like Do you have any other questions I can help with?"
    ),
    tools=tools,
    agent=support_qa_agent,
)



# ── Crew Assembly & Execution ──────────────────────────────────────
crew = Crew(
    agents=[support_agent, support_qa_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=True,
    memory=False 
)

# # Sample inquiry input
# inputs = {
#     "customer": "AcmeHealth",
#     "person": "Jane Doe",
#     "inquiry": (
#         "What services does Sikka offer for healthcare providers? "
#     )
# }

# if __name__ == "__main__":
#     # Kick off the multi-agent workflow and print the result
#     result = crew.kickoff(inputs=inputs)
#     print(result)

# - What services does Sikka.ai offer for dental practices?
# - What is ONE API?
# - What are the key features of Sikka Prime?
# - Why choose Sikka.ai?
# - How many practice installations and patients are on the Sikka.ai platform?
# - Which healthcare verticals (industries) does Sikka.ai support?


