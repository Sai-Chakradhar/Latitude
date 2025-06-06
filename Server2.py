# math_server.py
from mcp.server.fastmcp import FastMCP
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from datetime import datetime
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
import re
import os
from pymongo import MongoClient
from uuid import uuid4
from datetime import datetime
import os
from openai import OpenAI
import instructor
from pydantic import BaseModel, Field
from typing import List
import instructor
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

BIOPORTAL_API_KEY = os.getenv("BIOPORTAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

mcp = FastMCP("Medical")

@mcp.tool()
def process_medical_note(raw_text: str) -> dict:
    """Process medical note text and generate FHIR Condition resource with SNOMED codes"""
    
    def get_snomed_code(term):
        url = "https://data.bioontology.org/search"
        params = {
            "q": term,
            "ontologies": "SNOMEDCT",
            "require_exact_match": "true"
        }
        headers = {
            "Authorization": f"apikey token={BIOPORTAL_API_KEY}"
        }

        response = requests.get(url, params=params, headers=headers)
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            return "Unkown"

        data = response.json()
        collection = data.get("collection", [])

        if not collection:
            print(f"No SNOMED code found for '{term}'")
            return "Unkown"

        result = collection[0]
        return {
            "code": result["@id"].split("/")[-1],
            "display": result.get("prefLabel", term)
        }

    condition_fields = {
        "code": "What is the ICD-10 code and display name for the patient's primary diagnosis (e.g., acute on chronic mechanical low back pain)? Return the code, the system URL (http://hl7.org/fhir/sid/icd-10), and display name, as well as a text label for the diagnosis.",
        "clinicalStatus": "What is the current clinical status of the condition (e.g., active, inactive, remission)? Give only the status so that it can be looked up in the snomed_map",
        "verificationStatus": "What is the verification status of the diagnosis (e.g., confirmed, provisional, differential)? Give only the output don't explain yourself",
        "severity": "How severe is the condition according to the clinical documentation (e.g., mild, moderate, severe)?Give only the severity so that it can be looked up in the snomed_map",
        "onsetDateTime": "What is the exact or approximate date when the patient's current episode began? Return only a date in YYYY-MM-DD format if known, or respond with 'not specified' if unclear.",
        "bodySite": "Which anatomical site or region is affected by the condition (e.g., thorax, lumbar spine, left knee)?Give only the bodySite so that it can be looked up in the snomed_map",
        "evidence": "From the clinical note, extract only one clinical finding, test result, or diagnostic term that serves as the most direct evidence for the condition. Return just the keyword or phrase (e.g., 'back pain', 'chest X-ray', 'fever') with no explanation or formatting. Do not elaborate or explain — only return a single, specific term that can be looked up in SNOMED. If multiple are mentioned, choose the most clinically relevant or first one stated. eg chest pain, lower back pain etc",
        "note": "What additional narrative details or context are provided about the condition?",
        "category": "From the clinical note, identify whether this condition is recorded as a 'diagnosis', 'problem-list-item'. Return only one of these terms without explanation.",
        "asserter": "Who made the diagnosis or asserted the condition (e.g., physician, nurse practitioner, or patient themselves)? Extract their name or role. Output just the name of person who did it",
        "patient": "Who is the patient mentioned in this note? Extract the patient's full name or patient ID if explicitly available. Just answer directly don't brag",
        "encounter": "Which medical encounter (e.g., outpatient visit, ER consultation, inpatient admission) is associated with this condition entry? Provide the encounter ID available.",
        "recordedDate": "On what date was the condition officially recorded in the medical record or EHR? Just give output in MM-DD-YY format",
        "id": "What is the unique ID or identifier for this condition entry?"
    }
    
    # Initialize LLM and RAG components
    llm = ChatOpenAI(temperature=0)
    prompt = hub.pull("rlm/rag-prompt")
    raw_doc = Document(page_content=raw_text, metadata={"source": "medical_note.txt"})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?"],
    )
    docs = text_splitter.split_documents([raw_doc])
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embedding_model)
    
    # Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    retrieved_chunks_per_field = {}
    def run_rag_with_logging(question, field_key):
        # Dictionary to store retrieved chunks
        docs = retriever.invoke(question)
        context = docs
        retrieved_chunks_per_field[field_key] = [doc.page_content for doc in docs][:4]
        prompt_input = prompt.invoke({"context": context, "question": question})
        llm_output = llm.invoke(prompt_input)
        parsed_output = StrOutputParser().invoke(llm_output)
        return parsed_output.strip()

    # Process all fields using RAG
    llm_outputs = {}
    for field, question in condition_fields.items():
        value = run_rag_with_logging(question, field)
        llm_outputs[field] = value
    # Fields that need SNOMED lookup
    fields_to_lookup = ["code", "evidence", "bodySite", "severity", "category"]

    # Build SNOMED map dynamically from llm_outputs
    snomed_map = {}
    for field in fields_to_lookup:
        if field in llm_outputs:
            snomed_result = get_snomed_code(llm_outputs[field])
            if snomed_result is not None:
                snomed_map[llm_outputs[field]] = snomed_result

    def snomed_lookup(text):
        # First try the dynamically built snomed_map
        result = snomed_map.get(text.lower())
        if result:
            return result
        
        # Fallback to minimal hardcoded map for common terms
        lookup = {
            'lumbar spine': {'code': '122496007', 'display': 'Lumbar spine structure'},
            'mild': {'code': '255604002', 'display': 'Mild'},
            'diagnosis': {'code': '439401001', 'display': 'Diagnosis'}
        }
        return lookup.get(text.lower())

    # Build FHIR Condition resource
    condition_fhir = {
        "resourceType": "Condition",
        "id": llm_outputs.get("id", "condition-1"),
        "subject": {
            "reference": "Patient/",
            "display": llm_outputs.get("patient", "Unknown Patient")
        },
        "encounter": {
            "reference": llm_outputs.get("encounter", "Encounter/unknown")
        },
        "asserter": {
            "reference": "Practitioner/",
            "display": llm_outputs.get("asserter", "Unknown Practitioner")
        },
        "recordedDate": llm_outputs.get("recordedDate", datetime.today().strftime("%Y-%m-%d")),
        "clinicalStatus": {"coding": [{"code": llm_outputs.get("clinicalStatus", "active")}]},
        "verificationStatus": {"coding": [{"code": llm_outputs.get("verificationStatus", "confirmed")}]},
        "onsetDateTime": llm_outputs.get("onsetDateTime", "not specified"),
        "note": [{"text": llm_outputs.get("note", "")}]
    }

    # Add SNOMED coded fields
    # Code
    if llm_outputs.get("code") and (code := snomed_lookup(llm_outputs["code"])):
        condition_fhir["code"] = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": code["code"],
                "display": code["display"]
            }],
            "text": llm_outputs["code"]
        }

    # Severity
    if llm_outputs.get("severity") and (severity := snomed_lookup(llm_outputs["severity"])):
        condition_fhir["severity"] = {
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": severity["code"],
                "display": severity["display"]
            }],
            "text": llm_outputs["severity"]
        }

    # Category
    if llm_outputs.get("category") and (category := snomed_lookup(llm_outputs["category"])):
        condition_fhir["category"] = [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": category["code"],
                "display": category["display"]
            }],
            "text": llm_outputs["category"]
        }]

    # Evidence
    if llm_outputs.get("evidence") and (evidence := snomed_lookup(llm_outputs["evidence"])):
        condition_fhir["evidence"] = [{
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": evidence["code"],
                    "display": evidence["display"]
                }],
                "text": llm_outputs["evidence"]
            }
        }]

    # Body Site
    if llm_outputs.get("bodySite") and (body_site := snomed_lookup(llm_outputs["bodySite"])):
        condition_fhir["bodySite"] = [{
            "coding": [{
                "system": "http://snomed.info/sct",
                "code": body_site["code"],
                "display": body_site["display"]
            }],
            "text": llm_outputs["bodySite"]
        }]

    # Add generated narrative text
    condition_fhir["text"] = {
        "status": "generated",
        "div": f"<div><p><b>Condition:</b> {llm_outputs.get('code', 'Unknown')}</p><p><b>Severity:</b> {llm_outputs.get('severity', 'Unknown')}</p></div>"
    }
    
    return {
        "fhir_condition": condition_fhir,
        "llm_outputs": llm_outputs,
        "snomed_map": retrieved_chunks_per_field
    }

@mcp.tool()
def policy_check(resources) -> dict:
    # Load JSON back into a Python dict
    with open("output.json", "r") as f:
        loaded_data = json.load(f)

    # Convert string keys back to integers (optional)
    loaded_data = {int(k): v for k, v in loaded_data.items()}
    # Step 1: Define the Pydantic model
    class SummaryChoice(BaseModel):
        """Return the index of the best-matching summary."""
        choice: int = Field(..., ge=1, le=5, description="A number from 1 to 5 indicating the best matching summary.")

    # Step 2: Define your prompt builder
    def get_summary_selection_prompt(condition_text: str, summary_dict: dict[int, str]) -> str:
        numbered_summaries = "\n".join([f"{i}. {summary_dict[i]}" for i in sorted(summary_dict)])
        return f"""
    You are a clinical AI assistant. Your task is to evaluate which one of the five document summaries best supports or describes the given FHIR Condition resource.
    Do NOT choose a summary that excludes or contradicts the condition.
    Prefer summaries that affirmatively support the diagnosis and treatment approach.
    Exclude summaries that limit coverage to unrelated conditions (e.g., radiculopathy if the patient has mechanical back pain).
    FHIR Condition:
    \"\"\"{condition_text}\"\"\"

    Summaries:
    {numbered_summaries}

    Reply ONLY with a number from 0 to 4 that best matches the FHIR condition.
    Respond with the field: `choice`
    """

    # Step 3: Initialize OpenAI client with instructor
    client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    # Step 4: Use the client to get structured output
    def select_best_summary(condition_text: str, summary_dict: dict[int, str]) -> int:
        prompt = get_summary_selection_prompt(condition_text, summary_dict)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_model=SummaryChoice
        )
        return response.choice
    best_choice = select_best_summary(resources, loaded_data)
    print(f" Best summary is #{best_choice}: {loaded_data[best_choice]}")

    # 1. Structured output schema
    class PolicyCheckResult(BaseModel):
        probability: float = Field(..., ge=0, le=1, description="Probability of approval under the policy")
        conditions_met: List[str]
        conditions_missing: List[str]
        verdict: str = Field(..., description="approve / deny / needs more info")
        criteria: List[str]

    # 2. Load the policy text by index
    def load_policy_text(folder_path: str, index_to_filename: dict[int, str], best_idx: int) -> str:
        policy_file = os.path.join(folder_path, index_to_filename[best_idx])
        with open(policy_file, "r") as f:
            return f.read()

    # 3. Generate analysis prompt
    def build_policy_prompt(policy_text: str, condition_fhir: dict) -> str:
        return f"""
    You are a clinical policy reviewer. Given the following medical policy and a FHIR Condition, determine how well the condition meets the policy criteria.

    --- FHIR Condition ---
    {condition_fhir}

    --- Medical Policy Document ---
    \"\"\"{policy_text}\"\"\"

    Evaluate and return:
    1. A probability (between 0 and 1) that the condition qualifies under the policy
    2. A list of conditions or criteria that are clearly met
    3. A list of criteria that are missing or unclear
    4. A final verdict: approve / deny / needs more info
    5. All the criterias extracted from policy document
    """

    # 4. Run the check via GPT-4o using Instructor
    def run_policy_check(folder_path: str, index_to_filename: dict[int, str], best_idx: int, condition_fhir: dict) -> PolicyCheckResult:
        policy_text = load_policy_text(folder_path, index_to_filename, best_idx)
        prompt = build_policy_prompt(policy_text, condition_fhir)

        client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_model=PolicyCheckResult
        )
        return result
    # Suppose best_idx = 4 was selected earlier
    index_to_filename = {
        0: "/Users/saichakradhar/Desktop/Sai_C3/latitudeh/OCR/ocr_output_Bariatric_Surgery.txt",
        1: "/Users/saichakradhar/Desktop/Sai_C3/latitudeh/OCR/ocr_output_Epidural-Steroid-Injections-for-Back-and-Neck-Pain.txt",
        2: "/Users/saichakradhar/Desktop/Sai_C3/latitudeh/OCR/ocr_output_habilitative_services_outpatient_rehabilitation.txt",  
        3: "/Users/saichakradhar/Desktop/Sai_C3/latitudeh/OCR/ocr_output_LCD_Epi.txt",
        4: "/Users/saichakradhar/Desktop/Sai_C3/latitudeh/OCR/ocr_output_LCD_Laproscopic.txt"
    }

    result = run_policy_check("path/to/policy_docs", index_to_filename, best_idx=best_choice, condition_fhir=resources)

    print(f"\n✅ Verdict: {result.verdict}")
    print(f"📊 Probability: {result.probability * 100:.1f}%")
    print("✔️ Conditions Met:", result.conditions_met)
    print("❌ Conditions Missing:", result.conditions_missing)
    return result

@mcp.tool()
def prior_auth_generator(raw_text: str):
    resourse_condition = process_medical_note(raw_text)
    policy_output = policy_check(resourse_condition)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    prompt = f"""
    You are a clinical decision support agent.

    Using the following FHIR Condition resource and the policy evaluation result, generate a structured prior authorization JSON object. This object should be suitable for submission to a payer or EHR system and must include the following:

    - Patient information (name, reference ID)
    - Provider information (asserter)
    - Diagnosis code and description
    - Procedure being requested (based on policy match)
    - Clinical justification summary
    - Date of request
    - Policy alignment summary (criteria met and unmet)
    - Decision confidence (probability)
    - Suggested CPT code (if identifiable from policy match)
    - Final verdict (approve, deny, needs more info)

    Ensure that the output follows JSON syntax and uses clear field names.

    FHIR Condition:
    {resourse_condition}

    Policy Output:
    {policy_output}
    """ 
    pa = llm.invoke(prompt)
    return pa

 
if __name__ == "__main__":
    mcp.run(transport="stdio")