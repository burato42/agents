from crewai import Agent, Crew
from crewai import Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv, dotenv_values 

load_dotenv()

# Create a search tool
search_tool = SerperDevTool()

# Define agents
venue_finder = Agent(
    role='Conference Venue Finder',
    goal='Find the best venue for the upcoming conference',
    backstory=(
        'You are an experienced event planner with a knack for finding the perfect venues. '
        'Your expertise ensures that all conference requirements are met efficiently.'
    ),
    verbose=True,
    tools=[search_tool]
)

# Define tasks
find_venue_task = Task(
    description=(
        "Conduct a thorough search to find the best venue for the upcoming "
        "conference. Consider factors such as capacity, location, amenities, "
        "and pricing. Use online resources and databases to gather comprehensive "
        "information."
    ),
    expected_output=(
        "A list of 5 potential venues with detailed information on capacity, "
        "location, amenities, pricing, and availability."
    ),
    agent=venue_finder
)


crew = Crew(
  agents=[venue_finder],
  tasks=[find_venue_task],
  verbose=True
)
output = crew.kickoff()
# Print the output
print(output)
# Convert the CrewOutput object to a Markdown string
markdown_output = output.raw
print(markdown_output)
# Save the Markdown output to a file
with open('venue_finder_output.md', 'w') as f:
    f.write(markdown_output)