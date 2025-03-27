import streamlit as st
import pandas as pd
import json
import os
from classifier_utils import (
    load_model_and_tokenizer,
    extract_text_from_pdf,
    extract_text_from_eml,
    ensemble_classify,
    classify_sub_request,
    extract_critical_info
)

# --- INIT ---
st.set_page_config(page_title="Gen AI Email Triage", layout="wide")
st.markdown("""
<div style='background-color:#B31B1B; padding: 1rem; text-align: center; color: white; border-radius: 5px;'>
    <h1 style='margin: 0;'>Wells Fargo | Email Triage Console</h1>
</div>
""", unsafe_allow_html=True)

# --- Load model, tokenizer, and label mapping ---
model, tokenizer, label_mapping = load_model_and_tokenizer()

# --- Load request type configuration ---
CONFIG_FILE = "request_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        REQUEST_TYPES = json.load(f)
else:
    REQUEST_TYPES = {}

# --- HEADER ---
st.markdown("""
    <div style='padding: 1rem 0;'>
        <h2 style='color:#b31b1b;'>📧 Gen AI Triage Console</h2>
        <p style='color:gray;'>Smart Email & Document Classifier for Request Routing</p>
    </div>
""", unsafe_allow_html=True)

# --- LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.header("🔍 Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "eml"])
    
    if uploaded_file:
        st.success(f"Uploaded: {uploaded_file.name}")
        process_now = st.button("🚀 Run Classification")
    else:
        st.warning("📎 Upload a file to begin triage.")

with col2:
    st.header("🧠 Triage Results")
    if uploaded_file and 'process_now' in locals() and process_now:
        import hashlib
        from difflib import SequenceMatcher

        def is_duplicate(text):
            new_hash = hashlib.md5(text.encode()).hexdigest()
            if os.path.exists("processed_hashes.txt"):
                with open("processed_hashes.txt", "r") as f:
                    hashes = f.read().splitlines()
                    for h in hashes:
                        if SequenceMatcher(None, h, new_hash).ratio() > 0.95:
                            return True
            with open("processed_hashes.txt", "a") as f:
                f.write(new_hash + "\n")
            return False
        if uploaded_file.name.endswith(".eml"):
            pdf_text = extract_text_from_eml(uploaded_file)
        else:
            pdf_text = extract_text_from_pdf(uploaded_file)

        if is_duplicate(pdf_text):
            st.warning("⚠️ This message appears to be a duplicate of a previously triaged email.")

        result = ensemble_classify(pdf_text, model, tokenizer, label_mapping, REQUEST_TYPES)
        sub_request = classify_sub_request(pdf_text, result["Primary Classification"], REQUEST_TYPES)

        if sub_request is None:
            st.warning("⚠️ Sub-request could not be classified. Check Gemini response or if sub-options are defined.")

        st.markdown("""
        <div style='border: 1px solid #ccc; padding: 1rem; border-radius: 10px;'>
            <h4>🗂️ <strong>Request Type:</strong> <span style='color:#b31b1b;'>{}</span></h4>
            <h5>📌 <strong>Sub-Request Type:</strong> <span style='color:darkblue;'>{}</span></h5>
            <p><strong>Confidence:</strong> {:.2f}</p>
        </div>
        """.format(
            result["Primary Classification"],
            sub_request or "(Not applicable)",
            result["Confidence"]
        ), unsafe_allow_html=True)

        info = extract_critical_info(pdf_text)
        with st.expander("📋 Extracted Key Info", expanded=True):
            if info:
                st.json(info)
            else:
                st.info("No structured information could be extracted.")

        with st.expander("🔬 Detailed Probabilities and Votes"):
            st.write("**All Probabilities:**")
            st.json(result.get("All Probabilities", {}))
            st.write("**Model Votes:**")
            st.json(result.get("Votes", []))

    elif uploaded_file:
        st.caption("Hit the button to classify and extract insights.")

st.divider()

# --- Editable config section ---
with st.expander("⚙️ Manage Request & Sub-Request Types"):
    st.markdown("Add or update request types and their sub-types.")
    with st.form("update_config"):
        new_request = st.text_input("New Request Type")
        new_subs = st.text_area("Sub-Request Types (comma-separated)")
        if st.form_submit_button("➕ Add / Update"):
            if new_request:
                REQUEST_TYPES[new_request] = [s.strip() for s in new_subs.split(",") if s.strip()]
                with open(CONFIG_FILE, "w") as f:
                    json.dump(REQUEST_TYPES, f, indent=2)
                st.success(f"Request type '{new_request}' updated.")
    if REQUEST_TYPES:
        st.markdown("### Current Configuration")
        config_rows = [(req, sub) for req, subs in REQUEST_TYPES.items() for sub in (subs or ["(None)"])]
        df_config = pd.DataFrame(config_rows, columns=["Request Type", "Sub-Request Type"])
        st.dataframe(df_config, use_container_width=True)
    else:
        st.info("No request types defined yet. Use the form above to add one.")
