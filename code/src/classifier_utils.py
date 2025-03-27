import torch
import numpy as np
import google.generativeai as genai
from transformers import BertTokenizerFast, BertForSequenceClassification
import os
import json
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from email import policy
from email.parser import BytesParser
load_dotenv()

# Load model, tokenizer, and label mapping
def load_model_and_tokenizer(model_dir="email_classifier_bert_v2", label_map_file="label_mapping.json"):
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    with open(label_map_file, "r") as f:
        label_mapping = json.load(f)
    return model, tokenizer, label_mapping

# Predict using BERT model
def predict_email_verbose(text, model, tokenizer, label_mapping):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=1).detach().cpu().numpy()[0]
    result = {
        "All Probabilities": {label_mapping[str(i)]: round(float(p), 3) for i, p in enumerate(probs)}
    }
    pred_index = int(np.argmax(probs))
    result["Prediction"] = label_mapping[str(pred_index)]
    result["Confidence"] = round(float(probs[pred_index]), 3)
    return result

# Gemini fallback classification
def classify_gemini(text, label_options):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Classify this email into one of the following request types: {label_options}.\n\nEmail:\n{text}\n\nClassification:"
    response = model.generate_content(prompt)
    return response.text.strip()

# Override based on keyword presence
def priority_override(text, predicted_label):
    text_lower = text.lower()
    if "repay under sofr" in text_lower or "we will remit" in text_lower or ("aba" in text_lower and "usd" in text_lower):
        return "Money Movement-Inbound"
    return predicted_label

def ensemble_classify(text, model, tokenizer, label_mapping, request_types):
    base = predict_email_verbose(text, model, tokenizer, label_mapping)
    base_prediction = base["Prediction"]
    base_conf = base["Confidence"]

    try:
        gemini_pred = classify_gemini(text, list(request_types.keys()))
    except:
        gemini_pred = None

    votes = [base_prediction]
    if gemini_pred:
        votes.append(gemini_pred)

    vote_counts = {label: votes.count(label) for label in set(votes)}
    majority = max(vote_counts, key=vote_counts.get)

    # 🔁 New decision logic
    if gemini_pred and gemini_pred not in label_mapping.values():
        final = gemini_pred
        print("🆕 Gemini-only type detected. Overriding model prediction.")
    elif base_conf < 0.5 and gemini_pred:
        final = gemini_pred
        print("⚠️ Low confidence fallback to Gemini.")
    else:
        final = majority

    final = priority_override(text, final)

    base["Primary Classification"] = final
    base["Votes"] = votes

    if gemini_pred and gemini_pred != final:
        base["Secondary Classification"] = gemini_pred
    elif base_prediction != final:
        base["Secondary Classification"] = base_prediction
    else:
        base["Secondary Classification"] = None

    return base



# Sub-request classifier using Gemini
def classify_sub_request(text, main_class, request_types):
    sub_options = request_types.get(main_class, [])
    if not sub_options:
        return None
    prompt = f"""Classify this email into one of the following sub-request types: {sub_options}.

Email:
{text}

Sub-Request Classification:"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("⚠️ Sub-request classification failed:", e)
        return None

# PDF text extractor
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()



def extract_text_from_eml(eml_file):
    msg = BytesParser(policy=policy.default).parse(eml_file)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_content()
    else:
        return msg.get_content()
    return ""

def extract_critical_info(text):
    import google.generativeai as genai
    import os
    import json

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    prompt = f"""
    Extract the following fields from the below email and return them as a valid JSON object:

    - Amount
    - Date
    - CUSIP
    - Reference ID
    - Borrower Name

    Email:
    \"\"\"
    {text}
    \"\"\"
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # fallback: parse simple key-value pairs
            lines = response.text.splitlines()
            info = {}
            for line in lines:
                if ":" in line:
                    key, val = line.split(":", 1)
                    info[key.strip()] = val.strip()
            return info
    except Exception as e:
        print("⚠️ Critical info extraction failed:", e)
        return {}
