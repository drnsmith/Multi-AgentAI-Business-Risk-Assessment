from crewai import Task
from agents.retriever import retriever_agent

retrieval_task = Task(
    description="Find all documents relevant to the user's question.",
    agent=retriever_agent,
    expected_output="A list of relevant document chunks"
)


