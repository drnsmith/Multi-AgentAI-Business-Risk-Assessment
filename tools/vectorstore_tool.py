# tools/vectorstore_tool.py
import sys
import os

# Add the crewai_tools path manually
CREWAI_TOOLS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "crewai-tools", "crewai_tools"))
if CREWAI_TOOLS_PATH not in sys.path:
    sys.path.insert(0, CREWAI_TOOLS_PATH)

from crewai_tools.tool import BaseTool


class RiskRetrieverTool(BaseTool):
    name = "Risk Retriever"
    description = "Retrieves relevant chunks from risk reports based on a query."

    def __init__(self):
        super().__init__()
        self.vectorstore = load_vectorstore()

    def _run(self, input: str) -> str:
        docs = self.vectorstore.similarity_search(input, k=5)
        return "\n\n".join(doc.page_content for doc in docs)
    async def _arun(self, input: str) -> str:
        # Asynchronous version of the run method
        docs = await self.vectorstore.asimilarity_search(input, k=5)
        return "\n\n".join(doc.page_content for doc in docs)        
    
    def _get_tool_info(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for in the risk reports."
                    }
                },
                "required": ["query"]
            }
        }           
    

# This tool is designed to be used within an agent that needs to retrieve information




