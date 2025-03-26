# ai_integration.py
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
MODEL_NAME = "gemini-pro"

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
else:
    print("Error: GOOGLE_API_KEY not found in environment variables.")

def analyze_text(text_content):
    """
    Analyzes the provided text content using the Gemini Pro model to identify
    intent, summarize, classify request types, and extract relevant data.

    Args:
        text_content (str): The text extracted from the email or document.

    Returns:
        dict: A dictionary containing the AI analysis results.
    """
    if not GOOGLE_API_KEY:
        return {'error': 'Google Gemini API key not configured.'}

    prompt = f"""You are an AI assistant tasked with analyzing text from emails and documents.
    Your goal is to understand the user's intent, provide a concise summary of the content,
    identify the type of request being made (if any), and extract specific, relevant information.

    Consider the following potential request types: Information Request, Service Request, Problem Report, Change Request, Facility Lender Share Adjustment, Payment Inquiry, Document Request, Update Request. If none of these fit, indicate "Other".

    Extract key entities and data points relevant to the identified intent and request type.

    Format your response as a JSON object with the following keys:
    - "primary_intent": A brief description of the main purpose of the text.
    - "summary": A concise summary of the text's content.
    - "request_types": A list of identified request types (can be multiple if applicable) with a "confidence" score (e.g., [{"request_type": "Information Request", "confidence": 0.9}, ...]). Include "Custom Match" if a configured custom request type is strongly indicated by keywords.
    - "extracted_data": A dictionary of key-value pairs representing the extracted information. Be specific and use clear keys (e.g., "order_number": "12345", "lender_share_adjusted": true, "payment_due_date": "YYYY-MM-DD").
    - "confidence_score": An overall confidence score (0.0 to 1.0) for the accuracy of your analysis.

    Analyze the following text:
    ```
    {text_content}
    ```

    Respond with a JSON object.
    """

    try:
        response = model.generate_content(prompt)
        response.raise_for_status()
        gemini_output = response.text

        try:
            analysis_results = json.loads(gemini_output)
            print("Gemini Analysis Results:", analysis_results)
            return analysis_results
        except json.JSONDecodeError as e:
            print(f"Error decoding Gemini JSON output: {e}")
            print("Raw Gemini Output:", gemini_output)
            return {
                'error': f'Failed to decode JSON from Gemini: {e}',
                'raw_output': gemini_output
            }

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return {'error': f'Error calling Gemini API: {e}'}

if __name__ == '__main__':
    test_text_share_adjustment = """Dear Sir / Madam, here are the updates with respect to the deal please note that the Lender Share has
    been adjusted and you are required to make the necessary payments . Please find the attached document
    for your reference . Thanks"""
    results_share_adjustment = analyze_text(test_text_share_adjustment)
    print("\nShare Adjustment Analysis:")
    print(json.dumps(results_share_adjustment, indent=4))

    test_text_order_update = """Dear team, could you please provide an update on the delivery timeline for order #XYZ-789?
    We need to inform the client. Thanks."""
    results_order_update = analyze_text(test_text_order_update)
    print("\nOrder Update Analysis:")
    print(json.dumps(results_order_update, indent=4))