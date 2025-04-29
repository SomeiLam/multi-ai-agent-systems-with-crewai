import json
from crewai import Agent, Crew, Task
import os
from utils import get_openai_api_key, get_serper_api_key
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, field_validator

openai_api_key = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = get_serper_api_key()


# Initialize the tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Agent 1: Venue Coordinator
venue_coordinator = Agent(
    role="Venue Coordinator",
    goal="Identify and book an appropriate venue "
    "based on event requirements",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "With a keen sense of space and "
        "understanding of event logistics, "
        "you excel at finding and securing "
        "the perfect venue that fits the event's theme, "
        "size, budget constraints, city, "
        "and provide accurate attendee capacity figures (number of people)."
    ),
    model_kwargs={"max_tokens": 300}
)

 # Agent 2: Logistics Manager
logistics_manager = Agent(
    role='Logistics Manager',
    goal=(
        "Manage all logistics for the event "
        "including catering and equipment"
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Organized and detail-oriented, "
        "you ensure that every logistical aspect of the event "
        "from catering to equipment setup "
        "is flawlessly executed to create a seamless experience."
    ),
    model_kwargs={"max_tokens": 150}
)

# Agent 3: Marketing and Communications Agent
marketing_communications_agent = Agent(
    role="Marketing and Communications Agent",
    goal="Effectively market the event and "
         "communicate with participants",
    tools=[search_tool, scrape_tool],
    verbose=True,
    backstory=(
        "Creative and communicative, "
        "you craft compelling messages and "
        "engage with potential attendees "
        "to maximize event exposure and participation."
    ),
    model_kwargs={"max_tokens": 500}
)

# Pydantic model for structured output, ensuring capacity is always a string
default_capacity = "Not specified"
class VenueDetails(BaseModel):
    name: str
    address: str
    capacity: str = default_capacity
    booking_status: str

    @field_validator('capacity', mode='before')
    def ensure_capacity_str(cls, v):
        # If capacity comes in as dict, serialize to JSON string
        if isinstance(v, dict):
            return json.dumps(v)
        # Otherwise, coerce to string
        return str(v)

venue_task = Task(
    description=(
        "Find the best venue in {event_city} for \"{event_topic}\". "
        "Only return one best venue that meets the criteria. "
        "Return a JSON object with: name, full address, capacity (must always be returned as a string in JSON; "
        "if numeric, wrap in quotes; if multiple capacities exist, summarize them concisely in a single string), "
        "and booking availability status."),
    expected_output="A JSON object matching VenueDetails"
                    "All the details of a specifically chosen"
                    "venue you found to accommodate the event.",
    # output_json=VenueDetails,
    # output_file="venue_details.json",  
      # Outputs the venue details as a JSON file
    agent=venue_coordinator
)

logistics_task = Task(
    description="Coordinate catering and "
                 "equipment for an event "
                 "with {expected_participants} participants "
                 "on {tentative_date}.",
    expected_output="Confirmation of all logistics arrangements "
                    "including catering and equipment setup.",
    # async_execution=True,
    agent=logistics_manager
)

marketing_task = Task(
    description=(
        "Using the results from Task 1 (venueDetails) and Task 2 (logistics confirmation), "
        "write a comprehensive marketing report for {event_topic} to engage {expected_participants} attendees. "
        "Include a breakdown of channels used, key tactics, and engagement outcomes in Markdown format.\n\n"
        "Then return a single JSON object (no code fences) with two keys:\n"
        "  \"venueDetails\": <the JSON object from Task 1>,\n"
        "  \"marketingReport\": <the full Markdown report as a string>\n\n"
        "Example structure:\n"
        "{{\n"
        "  \"venueDetails\": {{ /* name, address, capacity, booking_status */ }},\n"
        "  \"marketingReport\": \"# Marketing Activities\\n- ...\\n## Attendee Engagement\\n- ...\"\n"
        "}}"
        "All the venueDetails properties must be returned as strings in JSON. "
        "Return **only** a valid JSON object, fenced as ```json. " 
        "All line breaks in `marketingReport` must be encoded as `\n`.  "
        "Do not include any raw Markdown outside the JSON."
    ),
    expected_output="A JSON-formatted string with keys `venueDetails` and `marketingReport`",
    agent=marketing_communications_agent
)

event_management_crew = Crew(
    agents=[venue_coordinator, logistics_manager, marketing_communications_agent],
    tasks=[venue_task, logistics_task, marketing_task],
    verbose=True
)

event_details = {
    'event_topic': "Sikka AI Developer Summit",
    'event_description': (
        "A one-day, hands-on summit in San Jose bringing together software "
        "engineers, healthcare IT leaders, and practice management partners "
        "to dive deep into the Sikka ONE API, explore the latest DentalLLM "
        "and Insights features, and share best-practice integration patterns."
    ),
    'event_city': "San Jose, California",
    'tentative_date': "2025-07-20",
    'expected_participants': 150,
    'budget': 30000,
    'venue_type': "Hotel conference center with main ballroom and breakout rooms"
}

# import json
# from pprint import pprint
# from IPython.display import Markdown

# if __name__ == "__main__":
#   result = event_management_crew.kickoff(inputs=event_details)
#   with open('venue_details.json') as f:
#     data = json.load(f)
#     pprint(data)
#     Markdown("marketing_report.md")
