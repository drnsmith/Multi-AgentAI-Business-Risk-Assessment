from modules.data_ingestion import ingest_data
from agents.risk_assessment_agent import RiskAssessmentAgent

def main():
    print("Starting Multi-Agent AI System for Business Risk Assessment...")

    # Step 1: Ingest Data
    data = ingest_data()
    print("Data ingestion complete.")

    # Step 2: Initialise Risk Assessment Agent
    agent = RiskAssessmentAgent(data)
    agent.run_analysis()

    print("Risk assessment complete.")

if __name__ == "__main__":
    main()
