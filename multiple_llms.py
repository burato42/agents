import os
from pprint import pprint
import warnings
from dotenv import load_dotenv, dotenv_values

from crewai import Agent, Task, Process, Crew
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool
from crewai_tools.tools.serper_dev_tool.serper_dev_tool import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

warnings.filterwarnings('ignore')
os.environ['PIP_ROOT_USER_ACTION'] = 'ignore'
load_dotenv()

# Create a search tool
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Initialize the Gemini model using ChatGoogleGenerativeAI
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                verbose=True,
                                temperature=0.5,
                                google_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the GPT-4 model using ChatOpenAI
gpt = ChatOpenAI(model="gpt-4-0314",
                 verbose=True,
                 temperature=0.5)


# Data Researcher Agent using Gemini and SerperSearch
article_researcher = Agent(
    role="Senior Researcher",
    goal='Unccover ground breaking technologies in {topic}',
    verbose=True,
    backstory=(
        "Driven by curiosity, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could change"
        "the world."
    ),
    tools=[search_tool],
    llm=gemini,
    allow_delegation=True
)

# Article Writer Agent using GPT
article_writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  # memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  llm=gpt,
  allow_delegation=False
)

# Research Task
research_task = Task(
    description=(
        "Conduct a thorough analysis on the given {topic}."
        "Utilize SerperSearch for any necessary online research. "
        "Summarize key findings in a detailed report."
    ),
    expected_output='A detailed report on the data analysis with key insights.',
    tools=[search_tool],
    agent=article_researcher,
)

# Writing Task
writing_task = Task(
    description=(
        "Write an insightful article based on the data analysis report. "
        "The article should be clear, engaging, and easy to understand."
    ),
    expected_output='A 6-paragraph article summarizing the data insights.',
    agent=article_writer,
)

# Form the crew and define the process
crew = Crew(
    agents=[article_researcher, article_writer],
    tasks=[research_task, writing_task],
    process=Process.sequential
)

research_inputs = {
    'topic': 'The rise in global tempratures from 2018 onwards'
}

# Kick off the crew
result = crew.kickoff(inputs=research_inputs)
# Convert the CrewOutput object to a Markdown string
result_markdown = result.raw
pprint(result_markdown)
# Save the Markdown output to a file
with open('article_writing_output.md', 'w') as f:
    f.write(result_markdown)
