from crewai import Task
from agents.synthesiser import synthesiser_agent
from tasks.retrieval_task import retrieval_task

synthesis_task = Task(
    description="Generate a clear, concise risk summary based on the retrieved documents.",
    agent=synthesiser_agent,
    context=[retrieval_task],
    expected_output="Markdown-formatted summary of key investment risks"
)

