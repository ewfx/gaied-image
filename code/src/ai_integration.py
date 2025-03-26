# ai_integration.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
from PyPDF2 import PdfReader
from datasets import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import logging

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
MODEL_NAME = "gemini-pro"
MODEL_GEMINI_FLASH = "gemini-1.5-flash"

# Configure logging for ai_integration.py
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model_gemini_pro = genai.GenerativeModel(MODEL_NAME)
    model_gemini_flash = genai.GenerativeModel(MODEL_GEMINI_FLASH)
    logger.info("Gemini API configured successfully.")
else:
    logger.error("GOOGLE_API_KEY not found in environment variables. Gemini models will not be available.")
    model_gemini_pro = None
    model_gemini_flash = None

# ✅ STEP 3: CONFIGURE REQUEST TYPES
REQUEST_TYPES = {
    "Adjustment": [],
    "AU Transfer": [],
    "Closing Notice": ["Reallocation Fees", "Amendment Fees", "Reallocation Principal"],
    "Commitment Change": ["Cashless Roll", "Decrease", "Increase"],
    "Fee Payment": ["Ongoing Fee", "Letter of Credit Fee"],
    "Money Movement-Inbound": ["Principal", "Interest", "Principal + Interest", "Principal+Interest+Fee"],
    "Money Movement - Outbound": ["Timebound", "Foreign Currency"]
}
logger.info(f"REQUEST_TYPES loaded: {REQUEST_TYPES}")

# ✅ STEP 11: INFERENCE FUNCTION (BERT Model - Load your trained model here)
TOKENIZER_PATH = "email_classifier_bert"
MODEL_PATH = "email_classifier_bert"
tokenizer = None
bert_model = None
main_label_mapping = None

def load_bert_model():
    global tokenizer
    global bert_model
    global main_label_mapping
    if tokenizer is None:
        try:
            tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_PATH)
            bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
            main_label_mapping = {i: label for label, i in {v: k for k, v in enumerate(sorted(REQUEST_TYPES.keys()))}.items()}
            logger.info("BERT model and tokenizer loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading BERT model from {MODEL_PATH} or tokenizer from {TOKENIZER_PATH}: {e}")

def predict_email_bert(text):
    if tokenizer is None or bert_model is None or main_label_mapping is None:
        logger.warning("BERT model not loaded. Call load_bert_model() first.")
        return {"Prediction": "N/A", "Confidence": 0.0}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    bert_model.eval()
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]

    try:
        pred_index = np.argmax(probs)
        prediction = main_label_mapping.get(pred_index, "Unknown")
        confidence = round(probs[pred_index], 3)
        logger.info(f"BERT prediction for text '{text[:50]}...': {prediction} (Confidence: {confidence})")
        return {"Prediction": prediction, "Confidence": confidence}
    except Exception as e:
        logger.error(f"Error mapping BERT prediction for text '{text[:50]}...': {e}")
        return {"Prediction": "Error", "Confidence": 0.0}

# ✅ STEP 12: ENSEMBLE CLASSIFIER (WITHOUT HF - Modified for direct use)
def classify_gemini(text):
    if model_gemini_flash is None:
        logger.warning("Gemini API key not configured. Skipping Gemini classification.")
        return None
    try:
        prompt = f"Classify this email into: {list(REQUEST_TYPES.keys())}.\n\nEmail:\n{text}\n\nClassification:"
        response = model_gemini_flash.generate_content(prompt)
        prediction = response.text.strip()
        logger.info(f"Gemini prediction for text '{text[:50]}...': {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Gemini classification failed for text '{text[:50]}...': {e}")
        return None

def priority_override(text, predicted_label):
    text_lower = text.lower()
    if "repay under sofr" in text_lower or "we will remit" in text_lower or ("aba" in text_lower and "usd" in text_lower):
        override_label = "Money Movement-Inbound"
        logger.info(f"Priority override triggered for text '{text[:50]}...'. Overriding prediction '{predicted_label}' with '{override_label}'.")
        return override_label
    return predicted_label

def ensemble_classify(text):
    load_bert_model() # Ensure BERT model is loaded before prediction
    bert_result = predict_email_bert(text)
    base_prediction = bert_result["Prediction"]
    base_conf = bert_result["Confidence"]

    gemini_pred = classify_gemini(text)

    votes = [base_prediction]
    if gemini_pred:
        votes.append(gemini_pred)
    logger.info(f"Ensemble votes for text '{text[:50]}...': BERT: {base_prediction} (Conf: {base_conf}), Gemini: {gemini_pred}")

    vote_counts = {label: votes.count(label) for label in set(votes)}
    majority = max(vote_counts, key=vote_counts.get)

    final = majority
    if base_conf < 0.5 and gemini_pred:
        final = gemini_pred
        logger.warning(f"Low BERT confidence ({base_conf}) for text '{text[:50]}...'. Falling back to Gemini prediction: {final}")

    final = priority_override(text, final)
    logger.info(f"Final ensemble classification for text '{text[:50]}...': {final}")

    return {
        "Final Classification": final,
        "BERT Prediction": base_prediction,
        "BERT Confidence": base_conf,
        "Gemini Prediction": gemini_pred,
        "Votes": votes
    }

# ✅ STEP 15: SUB-REQUEST CLASSIFICATION
sub_request_mapping = {k: v for k, v in REQUEST_TYPES.items() if v}
logger.info(f"SUB_REQUEST_MAPPING loaded: {sub_request_mapping}")

def classify_sub_request(text, main_class):
    if model_gemini_flash is None:
        logger.warning("Gemini API key not configured for sub-request classification.")
        return None
    sub_options = sub_request_mapping.get(main_class, [])
    if not sub_options:
        logger.info(f"No sub-request options found for main class: {main_class}")
        return None
    prompt = f"""Classify this email into one of the following sub-request types: {sub_options}.

Email:
{text}

Sub-Request Classification:"""
    try:
        response = model_gemini_flash.generate_content(prompt)
        prediction = response.text.strip()
        logger.info(f"Gemini sub-request prediction for main class '{main_class}' and text '{text[:50]}...': {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Sub-request classification failed for main class '{main_class}' and text '{text[:50]}...': {e}")
        return None

def analyze_text(extracted_text):
    """
    Analyzes the provided extracted text to classify its request type
    and optionally its sub-request type using the ensemble method.

    Args:
        extracted_text (str): The text content extracted from the document.

    Returns:
        dict: A dictionary containing the analysis results, including
              the final classification and potential sub-classification.
    """
    if not extracted_text:
        logger.warning("No text provided for analysis.")
        return {'error': 'No text provided for analysis.'}

    logger.info(f"Analyzing extracted text: '{extracted_text[:100]}...'")
    classification_result = ensemble_classify(extracted_text)
    final_classification = classification_result.get("Final Classification")
    sub_classification = None

    if final_classification in sub_request_mapping:
        sub_classification = classify_sub_request(extracted_text, final_classification)
        classification_result["Sub-Request Classification"] = sub_classification
        logger.info(f"Sub-classification for '{final_classification}': {sub_classification}")
    else:
        logger.info(f"No sub-request classification available for main class: {final_classification}")

    logger.info(f"Analysis result: {classification_result}")
    return classification_result

if __name__ == '__main__':
    # Example usage simulating text already extracted
    sample_extracted_text_pdf = """This notice serves as confirmation of the facility closing effective June 15, 2024.
Final payment of USD 1,250,000 received. Loan closure completed on the aforementioned date. All dues cleared.
Final reallocation completed. Facility closed with ref ID CL-9876."""
    logger.info(f"Starting analysis for sample PDF text: '{sample_extracted_text_pdf[:100]}...'")
    analysis_result_pdf = analyze_text(sample_extracted_text_pdf)
    print("\nAnalysis Result for Extracted PDF Text:")
    print(json.dumps(analysis_result_pdf, indent=4))
    logger.info(f"Analysis result for sample PDF text: {analysis_result_pdf}")

    sample_extracted_text_txt = "Please process an AU Transfer of USD 100000 to account 12345678."
    logger.info(f"Starting analysis for sample TXT text: '{sample_extracted_text_txt[:100]}...'")
    analysis_result_txt = analyze_text(sample_extracted_text_txt)
    print("\nAnalysis Result for Extracted Text File Content:")
    print(json.dumps(analysis_result_txt, indent=4))
    logger.info(f"Analysis result for sample TXT text: {analysis_result_txt}")
