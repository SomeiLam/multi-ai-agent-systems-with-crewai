from crewai import Agent, Task, Crew
import os
from utils import get_openai_api_key
import argparse
from rich.console import Console
from rich.markdown import Markdown

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'

## ── parse the topic from the CLI ───────────────────────────────
parser = argparse.ArgumentParser(
  description="Run the research_write_article crew with a given topic."
)
parser.add_argument(
  "topic",
  help="The topic you want the agents to research and write about"
)

args = parser.parse_args()
console = Console()

planner = Agent(
  role="LinkedIn Content Planner",
  goal=(
      "Outline an engaging LinkedIn post for '{topic}', "
      "including a one-sentence hook, 3–5 mini-paragraphs or bullets, "
      "emoji/visual cues, and 3–5 relevant hashtags."
  ),
  backstory=(
      "You are a LinkedIn strategist. Your goal is to outline a post for '{topic}' "
      "that immediately hooks readers, breaks content into very short bullets or lines, "
      "and calls out where to place images/emojis and which 3–5 hashtags to include."
  ),
  allow_delegation=False,
  verbose=True
)

writer = Agent(
  role="LinkedIn Content Writer",
  goal=(
      "Write a concise, <1300-char LinkedIn post on '{topic}' "
      "with a strong opening hook, 1–2 line paragraphs, "
      "appropriate emojis, and a clear call-to-action."
  ),
  backstory=(
      "You are a LinkedIn copywriter. Given an outline, draft a concise post under 1,300 chars "
      "with a strong opening hook, 1–2 line paragraphs, a storytelling flair, "
      "appropriate emojis, hashtags, and a clear call-to-action."
  ),
  allow_delegation=False,
  verbose=True
)

editor = Agent(
  role="LinkedIn Post Editor",
  goal=(
      "Polish the draft to ensure it follows LinkedIn best practices: "
      "hook first, 1–2 line paragraphs, correct emoji placement, "
      "3–5 hashtags, and a compelling CTA."
  ),
  backstory=(
      "You are an engagement-focused editor. Refine the draft so it follows LinkedIn best practices: "
      "ensuring the hook, line breaks, emoji placement, hashtag count, and CTA are all optimized."
  ),
  allow_delegation=False,
  verbose=True
)

plan = Task(
  description=(
      "Generate a LinkedIn post outline for '{topic}' that includes:\n"
      "  1. A one-sentence hook to grab attention.\n"
      "  2. 3–5 bullet points or mini-paragraphs summarizing key messages.\n"
      "  3. Suggestions for 3–5 relevant hashtags.\n"
      "  4. Emoji cues or visual prompts where helpful.\n"
      "  5. A closing call-to-action (e.g., question, link invite, comment prompt)."
  ),
  expected_output="A markdown-style outline detailing each element:\n"
        "- Hook\n"
        "- Mini-paragraphs or bullets\n"
        "- Hashtags list\n"
        "- Emoji/visual notes\n"
        "- CTA suggestion",
  agent=planner,
)

write = Task(
  description=(
      "Using the outline, write the full LinkedIn post for '{topic}':\n"
        "- Begin with the hook.\n"
        "- Use 1–2 line paragraphs with blank lines between.\n"
        "- Sprinkle in emojis as noted.\n"
        "- Append the hashtags at the end.\n"
        "- Conclude with the CTA."
  ),
  expected_output="A ready-to-publish LinkedIn post (plain text) under 1,300 characters, "
        "complete with line breaks, emojis, hashtags, and CTA.",
  agent=writer,
)

edit = Task(
  description=("Proofread and polish the LinkedIn post:\n"
        "- Confirm the first line is a hook.\n"
        "- Ensure paragraphs are 1–2 lines only.\n"
        "- Validate emoji placement and hashtag count (3–5).\n"
        "- Check overall length and CTA clarity.\n"
        "- Refine tone for maximum engagement."),
  expected_output="A final LinkedIn post draft that perfectly follows best practices "
        "and is ready to copy-paste into LinkedIn.",
  agent=editor
)

crew = Crew(
  agents=[planner, writer, editor],
  tasks=[plan, write, edit],
  verbose=2
)

# Kick it off
if __name__ == "__main__":
  topic = args.topic
  result = crew.kickoff(inputs={"topic": topic})
  console.print(Markdown(result))

# Example topics to try:
# - "The Future of AI in Healthcare"
# - "How to Build a Personal Brand on LinkedIn"
# - "The Importance of Diversity in Tech"
# - "Remote Work Best Practices"
# - "The Rise of Sustainable Business Practices"
# - "How to Master Time Management"
# - "The Power of Emotional Intelligence in Leadership"
# - "The Impact of Blockchain on Finance"
# - "How to Foster Innovation in Your Team"