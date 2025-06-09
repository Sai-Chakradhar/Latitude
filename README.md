# Clinical AI Assistant — FHIR & Prior Authorization Automation

This project is an end-to-end **Clinical Decision Support** system that processes unstructured medical notes, generates **FHIR Condition resources**, performs **policy compliance checks**, and outputs structured **Prior Authorization (PA) requests**. Built with LangChain, Streamlit, and MCP tooling.

---

## Features

*  **LLM-based Clinical Note Understanding**
  Extracts structured fields (e.g., diagnosis, severity, body site) using GPT-4o and RAG with FAISS.

*  **FHIR Resource Generation**
  Converts raw text into a valid `Condition` resource using SNOMED and ICD-10 codes.

*  **Policy Matching Engine**
  Uses GPT to determine which payer policy best supports the patient's condition and extracts approval probability, missing criteria, and verdict.

*  **Prior Authorization Generator**
  Assembles structured JSON suitable for payer submission based on the FHIR and policy results.

*  **Streamlit Chat Interface**
  Interactive UI to ask questions and invoke tools using LangGraph and LangChain agent.

---

## Tech Stack

* **LLM:** OpenAI GPT-4o via `langchain-openai`
* **Embeddings & Vector DB:** OpenAI Embeddings + FAISS
* **Prompting Framework:** LangChain + LangGraph `react_agent`
* **Server Framework:** MCP (`fastmcp`, `ClientSession`, `@mcp.tool`)
* **UI:** Streamlit
* **Ontology:** SNOMED CT via BioPortal API
* **FHIR Compliance:** Manual FHIR construction (`Condition` resource)

---

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Set up virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set environment variables in `.env`**

   ```env
   OPENAI_API_KEY=your-openai-key
   BIOPORTAL_API_KEY=your-bioportal-key
   ```

4. **Run MCP server**

   ```bash
   mcp dev Server2.py
   ```

5. **Launch Streamlit app**

   ```bash
   streamlit run mcpclient.py
   ```

---

## Tool Overview

| Tool Name              | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `process_medical_note` | Extracts clinical entities and builds FHIR `Condition` resource  |
| `policy_check`         | Matches FHIR data to policies and evaluates approval probability |
| `prior_auth_generator` | Outputs structured PA JSON using both FHIR + policy match        |

---

## Project Structure

```
├── mcpclient.py                  # Streamlit frontend
├── Server2.py              # MCP server with all tools
├── OCR/                    # Folder with policy OCR output text files
├── .env                    # Secrets (API keys)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Example Output

**Prior Authorization JSON Sample**

```json
{
  "patient": "John Doe",
  "diagnosis": {
    "code": "M54.5",
    "description": "Low back pain"
  },
  "probability": 0.92,
  "verdict": "approve",
  "clinical_justification": "...",
  "policy_alignment": {
    "met": ["chronic pain > 4 weeks", "failed conservative treatment"],
    "missing": []
  }
}
```

---

## Credits

Built by Sai Chakradhar using LangChain, MCP, OpenAI, and clinical ontologies. Designed for use in clinical documentation automation, EHR integration, and prior auth workflows.

---

## 📃 License

This project is under `MIT` unless otherwise specified.
