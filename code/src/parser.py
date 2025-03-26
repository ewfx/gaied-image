import email
import os
from typing import List, Dict, Tuple
import pandas as pd
from io import BytesIO
from transformers import pipeline

# Document parsing libraries
import pypdf
from docx import Document

# --- Request Type Configuration ---
REQUEST_TYPES_CONFIG = {
    "Adjustment": {
        "sub_requests": [],
        "keywords": ["adjust", "adjustment", "change", "modify"],
        "fields_to_extract": ["Deal Name", "Amount", "Effective Date", "Reason"]
    },
    "AU Transfer": {
        "sub_requests": [],
        "keywords": ["au transfer", "ownership transfer"],
        "fields_to_extract": ["Deal Name", "Transfer To", "Effective Date"]
    },
    "Closing Notice": {
        "sub_requests": ["Reallocation Fees", "Amendment Fees", "Reallocation Principal"],
        "keywords": ["closing notice", "loan closure", "payoff"],
        "fields_to_extract": ["Deal Name", "Closing Date"],
        "sub_request_keywords": {
            "Reallocation Fees": ["reallocation fee"],
            "Amendment Fees": ["amendment fee"],
            "Reallocation Principal": ["reallocation principal"]
        },
        "sub_request_fields": {
            "Reallocation Fees": ["Fee Amount", "Reallocation Date"],
            "Amendment Fees": ["Fee Amount", "Amendment Date"],
            "Reallocation Principal": ["Principal Amount", "Reallocation Date"]
        }
    },
    "Commitment Change": {
        "sub_requests": ["Cashless Roll", "Decrease", "Increase"],
        "keywords": ["commitment change", "facility change", "loan amount change"],
        "fields_to_extract": ["Deal Name", "Effective Date", "Old Commitment Amount", "New Commitment Amount"],
        "sub_request_keywords": {
            "Cashless Roll": ["cashless roll", "rollover"],
            "Decrease": ["decrease commitment", "reduce commitment"],
            "Increase": ["increase commitment", "add commitment"]
        },
        "sub_request_fields": {
            "Cashless Roll": ["Roll Over Date"],
            "Decrease": ["Decrease Amount"],
            "Increase": ["Increase Amount"]
        }
    },
    "Fee Payment": {
        "sub_requests": ["Ongoing Fee", "Letter of Credit Fee"],
        "keywords": ["fee payment", "pay fee", "fee invoice"],
        "fields_to_extract": ["Deal Name", "Payment Date", "Fee Type", "Amount"],
        "sub_request_keywords": {
            "Ongoing Fee": ["ongoing fee", "management fee"],
            "Letter of Credit Fee": ["letter of credit fee", "lc fee"]
        },
        "sub_request_fields": {
            "Ongoing Fee": ["Period Start Date", "Period End Date"],
            "Letter of Credit Fee": ["LC Number"]
        }
    },
    "Money Movement-Inbound": {
        "sub_requests": ["Principal Interest", "Principal + Interest", "Principal Interest+Fee"],
        "keywords": ["payment received", "funds in", "money in"],
        "fields_to_extract": ["Deal Name", "Transaction Date", "Amount"],
        "sub_request_keywords": {
            "Principal Interest": ["principal payment", "interest payment"],
            "Principal + Interest": ["principal and interest"],
            "Principal Interest+Fee": ["principal", "interest", "fee"]
        },
        "sub_request_fields": {
            "Principal Interest": ["Principal Amount", "Interest Amount"],
            "Principal + Interest": ["Total Amount"],
            "Principal Interest+Fee": ["Principal Amount", "Interest Amount", "Fee Amount", "Fee Type"]
        }
    },
    "Money Movement - Outbound": {
        "sub_requests": ["Timebound", "Foreign Currency"],
        "keywords": ["payment sent", "funds out", "money out"],
        "fields_to_extract": ["Deal Name", "Payment Date", "Amount", "Beneficiary"],
        "sub_request_keywords": {
            "Timebound": ["due date", "payment on"],
            "Foreign Currency": ["foreign currency", "exchange rate"]
        },
        "sub_request_fields": {
            "Timebound": ["Payment Due Date"],
            "Foreign Currency": ["Currency", "Exchange Rate"]
        }
    }
}

EXTRACTION_RULES = {
    "prioritize_body_for_request_type": True,
    "numerical_fields_preference": "attachments",  # Options: "body", "attachments", "both"
    "date_formats": ["YYYY-MM-DD", "MM/DD/YYYY", "DD-MMM-YYYY", "MMM DD, YYYY"] # More comprehensive date formats
}

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if False else -1) # Use GPU if available

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
    return text.strip()

def extract_text_from_docx(file_path: str) -> str:
    text = ""
    try:
        document = Document(file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path}: {e}")
    return text.strip()

def get_payload_text(payload):
    content_type = payload.get_content_type()
    content_disposition = str(payload.get("Content-Disposition"))

    if content_type == "text/plain" and "attachment" not in content_disposition:
        try:
            return payload.get_payload(decode=True).decode(payload.get_charset() or 'utf-8', errors='ignore').strip()
        except Exception:
            return ""
    elif payload.is_multipart():
        text_parts = []
        for sub_payload in payload.get_payload():
            text_part = get_payload_text(sub_payload)
            if text_part:
                text_parts.append(text_part)
        return "\n".join(text_parts)
    return ""

def extract_text_from_eml(file_path: str) -> Dict[str, str]:
    email_data = {"body": "", "attachments": {}}
    try:
        with open(file_path, 'rb') as file:
            msg = email.message_from_binary_file(file)

            if msg.is_multipart():
                body_parts = []
                for part in msg.walk():
                    if part.get_content_maintype() == 'text' and part.get('Content-Disposition') != 'attachment':
                        try:
                            body = part.get_payload(decode=True).decode(part.get_charset() or 'utf-8', errors='ignore').strip()
                            body_parts.append(body)
                        except Exception:
                            pass
                email_data["body"] = "\n".join(body_parts)
            else:
                email_data["body"] = get_payload_text(msg)

            for part in msg.walk():
                if part.get_content_maintype() not in ['text', 'multipart'] and part.get('Content-Disposition') is not None:
                    filename = part.get_filename()
                    if filename:
                        try:
                            payload = part.get_payload(decode=True)
                            if part.get_content_subtype() in ['plain', 'html']:
                                try:
                                    email_data["attachments"][filename] = payload.decode(part.get_charset() or 'utf-8', errors='ignore')
                                except UnicodeDecodeError:
                                    email_data["attachments"][filename] = "Binary Attachment"
                            else:
                                email_data["attachments"][filename] = "Binary Attachment"
                        except Exception as e:
                            email_data["attachments"][filename] = f"Error decoding attachment {filename}: {e}"

    except Exception as e:
        print(f"Error reading EML {file_path}: {e}")
    return email_data

def load_and_preprocess_email(file_path: str) -> Dict[str, str]:
    file_extension = os.path.splitext(file_path)[1].lower()
    email_content = {"body": "", "attachments": {}}

    if file_extension == '.pdf':
        email_content["body"] = extract_text_from_pdf(file_path)
    elif file_extension in ['.doc', '.docx']:
        email_content["body"] = extract_text_from_docx(file_path)
    elif file_extension == '.eml':
        email_content = extract_text_from_eml(file_path)
    else:
        print(f"Unsupported file format: {file_path}")
        return None

    return email_content

def classify_email(email_content: Dict[str, str], request_types_config: Dict, extraction_rules: Dict) -> List[Dict]:
    """Classifies the email content into request types using zero-shot classification."""
    body = email_content.get("body", "")
    attachment_text = "\n".join(email_content.get("attachments", {}).values())
    full_text = body if extraction_rules.get("prioritize_body_for_request_type", True) else f"{body}\n{attachment_text}"

    if not full_text.strip():
        return []

    candidate_labels = list(request_types_config.keys())
    results = zero_shot_classifier(full_text, candidate_labels, multi_label=True)

    classification_results = []
    for i, label in enumerate(results['labels']):
        confidence = results['scores'][i] * 100
        sub_classifications = []
        if label in request_types_config and request_types_config[label].get("sub_requests"):
            sub_labels = request_types_config[label]["sub_requests"]
            sub_results = zero_shot_classifier(full_text, sub_labels, multi_label=False)
            if sub_results['labels']:  # Check if any sub-classification was found
                sub_classifications.append({"sub_request_type": sub_results['labels'][0], "confidence": sub_results['scores'][0] * 100})

        classification_results.append({
            "request_type": label,
            "confidence": confidence,
            "sub_classifications": sub_classifications
        })

    classification_results.sort(key=lambda x: x['confidence'], reverse=True)
    return classification_results

def extract_data(email_content: Dict[str, str], classification_results: List[Dict], request_types_config: Dict, extraction_rules: Dict) -> Dict:
    """Extracts key information based on classification results and configuration."""
    extracted_data = {}
    body = email_content.get("body", "")
    attachments_text = "\n".join(email_content.get("attachments", {}).values())
    full_text = f"{body}\n{attachments_text}"

    for result in classification_results:
        req_type = result["request_type"]
        confidence = result["confidence"]
        fields_to_extract = request_types_config.get(req_type, {}).get("fields_to_extract", [])
        extracted_data[req_type] = {"confidence": confidence, "extracted_fields": {}}

        for field in fields_to_extract:
            # Basic keyword search for now - will be enhanced with LLMs
            if field.lower().replace(" ", "") in full_text.lower().replace(" ", ""):
                # Placeholder value - actual extraction needed
                extracted_data[req_type]["extracted_fields"][field] = "Found in text (needs proper extraction)"

        for sub_result in result.get("sub_classifications", []):
            sub_req_type = sub_result["sub_request_type"]
            sub_confidence = sub_result["confidence"]
            sub_fields_to_extract = request_types_config.get(req_type, {}).get("sub_request_fields", {}).get(sub_req_type, [])
            if req_type not in extracted_data:
                extracted_data[req_type] = {"confidence": confidence, "extracted_fields": {}} # Ensure base request type exists
            if sub_req_type not in extracted_data[req_type]:
                extracted_data[req_type][sub_req_type] = {"confidence": sub_confidence, "extracted_fields": {}}
            for field in sub_fields_to_extract:
                if field.lower().replace(" ", "") in full_text.lower().replace(" ", ""):
                    extracted_data[req_type][sub_req_type]["extracted_fields"][field] = "Found in text (needs proper extraction)"

    return extracted_data

def get_request_types_config():
    return REQUEST_TYPES_CONFIG

def get_extraction_rules():
    return EXTRACTION_RULES

if __name__ == "__main__":
    # Create a dummy EML file for testing
    with open("sample_email.eml", "w") as f:
        f.write("""From: sender@example.com
To: recipient@example.com
Subject: Commitment Increase for Deal ABC-123

Dear Recipient,

This email confirms an increase in the commitment amount for Deal ABC-123. The new commitment amount is $1,500,000, effective from 2025-03-28.

Sincerely,
Sender
""")

    email_content = load_and_preprocess_email("sample_email.eml")
    print("EML Content:", email_content)

    if email_content:
        rules = get_extraction_rules()
        config = get_request_types_config()
        classification_results = classify_email(email_content, config, rules)
        print("\nClassification Results:", classification_results)

        extracted_info = extract_data(email_content, classification_results, config, rules)
        print("\nExtracted Data:", extracted_info)

    os.remove("sample_email.eml")