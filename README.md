# Multi-Agent AI System for Business Risk Assessment

This project develops a Multi-Agent AI System designed to evaluate and monitor business risks by analysing diverse data sources, providing real-time insights, and facilitating proactive decision-making.

---

## Introduction
In today's dynamic business environment, organisations face various risks that can impact their operations and profitability. This system leverages advanced AI frameworks to provide comprehensive risk assessments, enabling businesses to make informed decisions and mitigate potential threats.

## Features

- **Data Ingestion and Indexing:** Collect and structure data from multiple sources, including financial reports, market analyses, and news articles.
- **Natural Language Processing:** Interpret and understand complex textual information relevant to various risk factors.
- **Multi-Agent Coordination:** Deploy specialised AI agents to analyse different aspects of business risk, such as financial, compliance, and market risks.
- **Information Retrieval:** Handle complex queries related to risk assessment, providing precise and relevant information.
- **Performance Evaluation:** Assess the system's outputs to ensure accuracy, relevance, and reliability in risk assessments.

---
## Architecture
The system comprises the following components:

1. **Data Ingestion and Indexing:**
   - Utilises **LlamaIndex** to ingest and structure data from various sources, facilitating efficient indexing and retrieval.

2. **Natural Language Processing:**
   - Employs LangChain to process and understand the ingested data, enabling interpretation of complex textual information pertinent to risk factors.

3. **Multi-Agent Coordination:**
   - Deploys **AutoGen** to orchestrate multiple AI agents, each specialising in different aspects of risk assessment, ensuring a cohesive analysis.

4. **Information Retrieval:**
   - Integrates **Haystack** to handle complex queries related to risk assessment, retrieving precise information in response to specific questions.

5. **Performance Evaluation:**
   - Employs **DeepEval** to assess the performance of the AI agents and the overall system, ensuring trustworthy outputs.

---
## Installation
To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/drnsmith/Multi-AgentAI-Business-Risk-Assessment.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Multi-AgentAI-Business-Risk-Assessment
   ```

3. **Create a virtual environment:**

   ```bash
   python3 -m venv venv
   ```

4. **Activate the virtual environment:**

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS/Linux:

     ```bash
     source venv/bin/activate
     ```

5. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
----
## Usage
After installation, you can run the system using the following command:

```bash
python main.py
```

This will start the AI agents and begin the risk assessment process based on the configured data sources.

---
## Contributing
We welcome contributions to enhance the functionality and performance of this system. Please fork the repository and submit a pull request with your proposed changes.

---
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


