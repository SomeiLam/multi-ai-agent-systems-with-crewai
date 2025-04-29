from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool, ScrapeWebsiteTool
import os
from pathlib import Path
from utils import get_openai_api_key, get_serper_api_key

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4-turbo'
os.environ["SERPER_API_KEY"] = get_serper_api_key()

# —───────────── Define Custom Tools ─────────────—
class SentimentAnalysisTool(BaseTool):
  name: str ="Sentiment Analysis Tool"
  description: str = ("Checks tone to ensure the email is "
        "positive, professional, and engaging.")
  
  def _run(self, text: str) -> str:
      # Your custom code tool goes here
      return "positive"
    
sentiment_tool = SentimentAnalysisTool()

# Tools to read your instruction files and templates
# directory_read_tool = DirectoryReadTool(directory='./instructions')

HERE = Path(__file__).parent
directory_read_tool = DirectoryReadTool(
    directory=str(HERE / "instructions")
)
file_read_tool = FileReadTool()

# Scraper that pulls content from all the relevant Sikka pages
scrape_tool = ScrapeWebsiteTool(
    name="Sikka Website Scraper",
    description="Fetches and summarizes content from Sikka’s key pages to inform email personalization.",
    urls=[
        "https://www.sikka.ai/",
        "https://www.sikka.ai/about-us",
        "https://www.sikka.ai/oneapi",
        "https://www.sikkasoft.com/optimizer",
        "https://www.sikka.ai/fee-survey",
        "https://www.sikka.ai/sikka-prime"
    ]
)

search_tool = SerperDevTool()

# ────────────── Agents ──────────────
prospect_profiling_agent = Agent(
    role="Business Development Intelligence Analyst",
    goal=(
        "Identify and validate key company and decision-maker information "
        "for targeted outreach; when specifics are unavailable, "
        "provide relevant industry context without guessing."
    ),
    backstory=(
        "You are a Business Development Intelligence Analyst on Sikka.ai’s Growth team. "
        "Your mission is to find and verify client company data, recent news, "
        "and leadership backgrounds using only trusted public sources and premium databases. "
        "You rigorously cross-check multiple sources to ensure every detail is accurate—no assumptions. "
        "If a specific data point cannot be confirmed, you default to providing general industry "
        "insights or benchmarks to maintain credibility."
    ),

    allow_delegation=False,
    verbose=True
)

sikka_web_analysis_agent = Agent(
    role="Market Intelligence Analyst",
    goal="Extract, validate, and structure Sikka.ai’s online product and brand intelligence to inform our outreach strategy.",
    backstory=(
        "You are a Senior Market Intelligence Analyst on the Strategy & Insights team at Sikka.ai. "
        "Charged with deep-diving into our digital footprint, you leverage advanced web‐scraping tools and research "
        "methodologies to compile authoritative summaries of our product offerings, customer value propositions, "
        "and competitive differentiators. Your work ensures every customer-facing message is grounded in the "
        "latest, most accurate intelligence."
    ),
    allow_delegation=False,
    verbose=True
)

email_agent = Agent(
    role="Marketing Communications Manager",
    goal="Craft data-driven, personalized email campaigns that align Sikka.ai’s value proposition with each prospect’s needs.",
    backstory=(
        "As Marketing Communications Manager at Sikka.ai, you translate strategic insights into compelling narratives. "
        "Partnering closely with Market Intelligence and Sales teams, you use our official email templates and the "
        "latest prospect profiles to write messages that resonate—balancing technical detail with a human touch. "
        "You uphold brand consistency, apply best‐practice frameworks, and optimize for engagement and conversion."
    ),
    allow_delegation=False,
    verbose=True
)

# ────────────── Tasks ──────────────

# 1) Profile the prospect via search
customer_profiling_task = Task(
    description=(
        "Conduct a comprehensive research dossier on {lead_name}, a {industry} organization evaluating dental technology solutions. "
        "Use SerperDevTool to verify company fundamentals, leadership bios, recent news or milestones, and market positioning. "
        "Determine the practice’s scale—classify as `solo`, `group`, or `enterprise`—based on number of providers, locations, or other available metrics. "
        "This dossier will underpin our personalized outreach strategy.\n"
        "Do not speculate—only include details you can confirm from trusted sources; if a fact isn’t verifiable, provide general industry context instead."
    ),
    expected_output=(
        "A detailed prospect dossier for {lead_name}, including:\n"
        "• Company overview and mission statement\n"
        "• Key decision-makers’ names, roles, and backgrounds\n"
        "• Recent initiatives, news, or growth milestones\n"
        "• Identified pain-points or challenges in the {industry}\n"
        "• Classification of practice size (`solo`, `group`, or `enterprise`) with supporting rationale\n"
        "• Opportunities where Sikka.ai’s OneAPI, Optimizer, or Fee Survey can add value\n"
        "• Recommended initial messaging angles"
    ),
    tools=[search_tool],
    agent=prospect_profiling_agent,
)

# 2) Summarize Sikka.ai
sikka_analysis_task = Task(
    description=(
        "Compile a detailed profile of Sikka.ai by scraping their key pages. "
        "Focus on product features, target customers, brand values, and recent initiatives."
    ),
    expected_output=(
        "A structured Sikka.ai summary:\n"
        "• Core offerings & features\n"
        "• Customer pain-points addressed\n"
        "• Brand & culture highlights\n"
        "• Mapping of each URL to its key insights"
    ),
    tools=[scrape_tool],
    agent=sikka_web_analysis_agent
)

# 3) Generate the actual outreach emails
sikka_outreach_task = Task(
    description=(
        "Based on the prospect dossier and inferred practice size (`solo`, `group`, or `enterprise`):\n"
        "1. Load the matching template file from `./instructions`:\n"
        "   - `solo_practice.txt` for `solo`\n"
        "   - `group_practice.txt` for `group`\n"
        "   - `enterprise_practice.txt` for `enterprise`\n"
        "2. Then, using that template plus:\n"
        "  • Prospect dossier (company overview, key decision-makers, recent milestones)\n"
        "  • Sikka.ai summary (from sikka_analysis_task)\n"
        "  • Inputs: {lead_name}, {industry}, {recipient_name}, {recipient_position}, {recent_event}, {core_feature}\n"
        "3. Populate every placeholder in the template. Ensure each draft:\n"
        "  - Opens with a personalized reference to {recent_event} or a dossier insight\n"
        "  - Weaves in Sikka’s core value proposition in context of the inferred practice size\n"
        "  - Preserves all section headings and structure from the template file\n"
        "  - Concludes with a clear, role-appropriate call to action\n"
        "  - Passes through SentimentAnalysisTool for a positive, on-brand tone"
        "**Return** exactly one Markdown string (no JSON, no code fences, no triple backticks, not in a code block) "
        "containing the fully populated email."
    ),
    expected_output=(
        "1 fully populated email drafts, matching the selected template’s structure, including:\n"
        "• A customized subject line\n"
        "• All template sections filled with tailored content\n"
        "• Confirmation that tone checks passed"
    ),
    tools=[directory_read_tool, file_read_tool, sentiment_tool],
    agent=email_agent,
)

# ────────────── Crew & Helper ──────────────
outreach_crew = Crew(
    agents=[
        prospect_profiling_agent,
        sikka_web_analysis_agent,
        email_agent
    ],
    tasks=[customer_profiling_task, sikka_analysis_task, sikka_outreach_task],
    verbose=True,
    memory=True
)

# def generate_personalized_email(
#     lead_name: str,
#     industry: str,
#     recipient_name: str,
#     recipient_position: str,
#     recent_event: str,
#     core_feature: str = None
# ) -> str:
#     """
#     1) Profiles the prospect
#     2) Summarizes Sikka.ai
#     3) Generates 1 personalized email drafts
    
#     Returns the raw markdown of the drafts.
#     """
#     inputs = {
#         "lead_name": lead_name,
#         "industry": industry,
#         "recipient_name": recipient_name,
#         "recipient_position": recipient_position,
#         "recent_event": recent_event,
#         "core_feature": core_feature or "OneAPI data integration"
#     }
#     result = crew.kickoff(inputs=inputs)
#     return result.raw

# ────────────── Example Usage ──────────────
# if __name__ == "__main__":
#     emails_md = generate_personalized_email(
#         lead_name="Maple Grove Dental",
#         industry="Dental Practice Management",
#         recipient_name="Dr. S. Christopher Chang",
#         recipient_position="Practice Owner",
#         recent_event="your recent fee survey results"
#     )
#     print(emails_md)

# ────────────── Example Output ──────────────
# ```text
# Subject: Elevate Maple Grove Dental’s Performance with Sikka OneAPI

# Dear Dr. Jim Tauschek,

# I hope this message finds you well. Your dedication to providing a wide range of dental services from general dentistry to advanced orthodontics at Maple Grove Dental is truly impressive. We recently came across your excellent survey results and think that there’s a great match between your technological aspirations and our solutions.

# At Sikka.ai, we recognize that managing such a diverse and technologically advanced practice is no small feat. That's where our OneAPI platform can play a pivotal role. It’s designed to seamlessly connect all your technological systems, ensuring that data flows smoothly between your operations without any hitches.

# Our Fee Survey and Sikka Prime analytics can further help you fine-tune your services’ pricing structure to stay competitive in Madison’s dynamic market. Coupling this with our Optimizer, which automates routine billing and administrative tasks, could free up your team to focus even more on patient care and less on paperwork.

# How about we schedule a quick 20-minute call to discuss how these tools can specifically benefit Maple Grove Dental? I’d love to show you how we can assist in streamlining your operations and enhancing your service delivery.

# Looking forward to your thoughts.

# Best regards,
# [Your Name]
# Marketing Communications Manager, Sikka.ai
# ```