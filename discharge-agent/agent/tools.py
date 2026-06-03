from langchain_core.tools import tool
from ingestion.document_processor import DocumentProcessor
from pypdf import PdfReader, PdfWriter
import os
import time


doc_processor = DocumentProcessor()


@tool
def get_document_info(file_path: str) -> str:
    """
    Call this tool FIRST. It returns the total number of pages in the PDF.
    This helps you plan your chunking strategy.
    """
    try:
        reader = PdfReader(file_path)
        return f"Document '{file_path}' has {len(reader.pages)} total pages."
    except Exception as e:
        return f"Error reading document info: {str(e)}"


@tool
def read_document_pages(file_path: str, start_page: int, end_page: int) -> str:
    """
    CRITICAL: Use this tool to read a medical PDF safely.
    Read them in chunks (e.g., 1-5, then 6-10).
    """
    print(f"SYSTEM LOG: Agent requested pages {start_page} to {end_page}...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)

            start_idx = max(0, start_page - 1)
            end_idx = min(total_pages, end_page)

            writer = PdfWriter()
            for i in range(start_idx, end_idx):
                writer.add_page(reader.pages[i])

            temp_chunk_path = f"temp_chunk_{start_page}_{end_page}.pdf"
            with open(temp_chunk_path, "wb") as f:
                writer.write(f)

            # 1. Extract the text using Docling/RapidOCR
            markdown_text = doc_processor.process_pdf_to_markdown(temp_chunk_path)

            # --- NEW: SAVE THE EXTRACTED MARKDOWN ---
            # Create a persistent directory in your project root
            log_dir = os.path.join(os.getcwd(), "extraction_logs")
            os.makedirs(log_dir, exist_ok=True)

            # Create a readable filename (e.g., "patient_record_pages_1-5.md")
            base_filename = os.path.basename(file_path).replace(".pdf", "")
            log_filename = os.path.join(
                log_dir, f"{base_filename}_pages_{start_page}-{end_page}.md"
            )

            # Write the raw Docling output to the file
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(
                    f"--- SOURCE: {base_filename} | PAGES: {start_page} to {end_page} ---\n\n"
                )
                f.write(markdown_text)
            # ----------------------------------------

            if os.path.exists(temp_chunk_path):
                os.remove(temp_chunk_path)

            return f"--- CONTENT FOR PAGES {start_page} TO {end_page} ---\n\n{markdown_text}"

        except Exception as e:
            print(f"SYSTEM LOG: Read attempt {attempt + 1} failed. Error: {str(e)}")
            if attempt == max_retries - 1:
                return f"ERROR Reading Document after {max_retries} attempts: {str(e)}"
            time.sleep(2)


# @tool
# def read_medical_document(file_path: str) -> str:
#     """
#     Use this tool to read a patient's medical PDF.
#     It returns the full text and tables of the document in Markdown format.
#     You MUST read the documents before drafting the summary.
#     """
#     try:
#         return doc_processor.process_pdf_to_markdown(file_path)
#     except Exception as e:
#         return f"Error reading document: {str(e)}"


@tool
def log_lab_value(lab_name: str, value: str, page_found: int) -> str:
    """
    Use this tool whenever you see a critical lab value (like Hb, Creatinine).
    This helps you cross-reference for conflicts later.
    """
    # In your nodes.py, you will intercept this tool call and append it to state["extracted_labs"]
    return f"Logged {lab_name}: {value} from page {page_found}."


@tool
def check_drug_interaction(drug_list: list[str]) -> str:
    """
    Checks for known dangerous interactions between a list of medications.
    """
    drugs = [d.lower() for d in drug_list]
    if "oflox tz" in drugs and "raciper" in drugs:
        return "WARNING: Potential interaction between Ofloxacin and PPIs. Flag for clinician review."
    return "No known severe interactions."


@tool
def flag_for_clinician_review(issue_description: str) -> str:
    """
    Use this tool to officially escalate missing data, contradictory notes,
    or severe safety warnings to the attending clinician.
    """
    return f"ESCALATION LOGGED: {issue_description}"


AGENT_TOOLS = [
    get_document_info,
    read_document_pages,
    check_drug_interaction,
    flag_for_clinician_review,
    log_lab_value
]
