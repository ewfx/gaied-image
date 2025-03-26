# Smart Email and Document Triage System

## Overview

This project implements a smart system for automatically classifying incoming emails and documents into predefined request types. It leverages a combination of Natural Language Processing (NLP) techniques, specifically a fine-tuned BERT model, and a Large Language Model (LLM), Google Gemini, to achieve robust and accurate classification. The system includes functionalities for API key testing, request type configuration, synthetic data generation, dataset preparation, model training and evaluation, ensemble classification, PDF text extraction, and sub-request classification.

This system aims to automate the manual triage process of customer communications, leading to faster processing times, reduced errors, and improved operational efficiency.

## Features

* **API Key Testing:** Ensures the availability and functionality of the Google Gemini API key.
* **Configurable Request Types:** Defines a structured schema for classifying incoming communications, including main request types and their associated sub-request types.
* **Synthetic Data Generation:** Augments the training dataset with realistic, automatically generated email examples for various request types, improving model generalization.
* **Comprehensive Data Preparation:** Includes loading datasets, handling duplicate entries, and encoding categorical labels into a numerical format suitable for machine learning models.
* **Fine-tuned BERT Model:** Utilizes a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model for text classification, fine-tuned on the prepared dataset to accurately categorize emails.
* **Evaluation Metrics:** Employs accuracy as the primary metric to evaluate the performance of the trained BERT model.
* **Ensemble Classification:** Combines the predictive power of the fine-tuned BERT model and Google Gemini to enhance classification accuracy and robustness.
* **Keyword-Based Override:** Implements a mechanism to override the ensemble prediction based on the presence of specific keywords in the email or document text, allowing for rule-based adjustments.
* **PDF Text Extraction:** Enables the system to process PDF documents by extracting their textual content for classification.
* **Sub-Request Classification (Optional):** Provides a secondary level of classification to identify more specific sub-request types within a main request category using Google Gemini.

## Setup

### Prerequisites

* **Python 3.x**
* **pip** (Python package installer)
* **Virtual Environment (recommended)**
* **Google Cloud Account and Gemini API Key:** You will need to obtain an API key from Google Cloud to use the Gemini models.
* **Initial Email Dataset:** The system expects an initial CSV file named `emails_dataset.csv` containing at least an `email_text` column and a `request_type` column.

### Installation

1.  **Clone the repository (if applicable) or create a new project directory.**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file with the following dependencies if it's not provided in the project)*
    ```
    PyPDF2
    datasets
    transformers
    evaluate
    torch
    google-generativeai
    pandas
    numpy
    ```

4.  **Set up the Google Gemini API Key:**
    * Go to the Google Cloud Console and obtain a Gemini API key.
    * **Securely set the API key as an environment variable.** It is strongly recommended **not** to hardcode the API key directly in the script.
        ```bash
        export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"  # On Linux/macOS
        set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"     # On Windows
        ```
        Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.

5.  **Prepare the initial email dataset:**
    * Ensure you have a CSV file named `emails_dataset.csv` in the project directory.
    * This file should contain at least two columns:
        * `email_text`: The content of the email or document.
        * `request_type`: The manually assigned category for the corresponding `email_text`.

## System Components and Functionality (Detailed)

The system operates through a series of well-defined steps:

1.  **✅ TEST API KEYS:**
    * **Mechanism:** The `test_gemini_api()` function initializes the Google Gemini API client using the API key retrieved from the environment variable (`GEMINI_API_KEY`). It then attempts to communicate with the Gemini API by sending a simple request to generate the phrase "Hello from Gemini!".
    * **Code:** The code uses the `google.generativeai` library to configure the API and interact with the `gemini-1.5-flash` model. A `try...except` block handles potential exceptions that might occur due to an invalid API key, network issues, or problems with the Gemini service.
    * **Reasoning:** This step is crucial for verifying that the system has the necessary credentials and can successfully connect to the Google Gemini API before proceeding with other operations that depend on it (like ensemble and sub-request classification).

2.  **✅ STEP 3: CONFIGURE REQUEST TYPES:**
    * **Mechanism:** The `REQUEST_TYPES` dictionary is a Python dictionary that explicitly defines the main categories (keys) into which emails and documents will be classified. The values associated with each main category are lists representing potential sub-categories.
    * **Code:** The `REQUEST_TYPES` dictionary is hardcoded in the script, providing a static configuration of the classification schema.
    * **Reasoning:** This configuration serves as the foundation for the entire classification process. It defines the target labels for the machine learning model and the categories used by the ensemble classifier (Gemini). The sub-request types enable a more granular level of categorization when needed. The image "Configure Request types" suggests that this dictionary would likely be the backend for a user interface allowing dynamic management of these request types.

3.  **✅ STEP 4: GENERATE SYNTHETIC DATA FIRST (EXPANDED FOR SUB-REQUESTS):**
    * **Mechanism:** The code employs a template-based approach to generate synthetic email text for various request types. For each request type, a list of predefined templates containing placeholders for dynamic information (e.g., dates, amounts, account numbers) is used. These placeholders are then filled with randomly generated values that mimic realistic data formats.
    * **Code:** The code defines separate lists of templates (`money_inbound_templates`, `adjustment_templates`, etc.) for different main request types. It then iterates a specific number of times for each type, randomly selecting a template and populating it with random data using the `format()` method. Each generated synthetic email and its corresponding `request_type` are stored in the `synthetic_data` list. Finally, this list is converted into a Pandas DataFrame and saved to `synthetic_emails.csv`.
    * **Reasoning:** Synthetic data generation is essential for several reasons:
        * **Augmenting Training Data:** It increases the size of the training dataset, which can lead to better model performance, especially for underrepresented classes in the real-world data.
        * **Improving Generalization:** By creating diverse examples based on templates, the model learns the underlying patterns of each request type rather than memorizing specific phrasings from the real dataset.
        * **Addressing Data Imbalance:** Synthetic data can be generated specifically for less frequent request types to balance the training data and prevent the model from being biased towards more common categories.

4.  **✅ STEP 5: LOAD DATASET AND PREP LABELS:**
    * **Mechanism:** This step involves loading the initial email dataset (`emails_dataset.csv`) and the generated synthetic dataset (`synthetic_emails.csv`) using the Pandas library. The two datasets are then concatenated into a single DataFrame. Duplicate email texts are removed to ensure the model learns from unique examples. Finally, the categorical `request_type` labels are converted into numerical labels using Pandas' categorical encoding. A mapping dictionary is created to store the correspondence between the numerical labels and the original request type names.
    * **Code:** The code uses `pd.read_csv()` to load the CSV files, `pd.concat()` to combine them, `df.drop_duplicates()` to remove duplicates based on the `email_text` column, and `astype("category").cat.codes` to perform label encoding. `dict(enumerate(...))` creates the `main_label_mapping`.
    * **Reasoning:** Machine learning models, particularly the BERT model used here, require numerical input. Therefore, converting the text-based request types into numerical labels is a necessary preprocessing step. Removing duplicates prevents the model from overfitting to identical examples. Combining the real and synthetic data creates a larger and more diverse training set.

5.  **✅ STEP 5: CONVERT TO DATASET AND TOKENIZE:**
    * **Mechanism:** The preprocessed Pandas DataFrame is converted into a Hugging Face `Dataset` object, which is a more efficient format for training with the Transformers library. A pre-trained tokenizer (`BertTokenizerFast`) from the `bert-base-uncased` model is loaded. A `tokenize_function` is defined to convert the raw email text into token IDs, attention masks, and token type IDs, which are the input format required by the BERT model. This function also handles padding and truncation to ensure all input sequences have a consistent length. The `tokenize_function` is then applied to the entire dataset using the `map()` function. Finally, the tokenized dataset is split into training and evaluation sets.
    * **Code:** The code uses `Dataset.from_pandas()` for conversion, `BertTokenizerFast.from_pretrained()` to load the tokenizer, and the `dataset.map()` function to apply the `tokenize_function`. `train_test_split()` is used to split the data.
    * **Reasoning:** The BERT model cannot directly process raw text. Tokenization is the process of breaking down the text into smaller units (tokens) and converting them into numerical IDs that the model can understand. Padding and truncation are essential for handling sequences of varying lengths. Splitting the data into training and evaluation sets allows for assessing the model's performance on unseen data.

6.  **✅ STEP 6: DEFINE MODEL:**
    * **Mechanism:** A pre-trained BERT model specifically designed for sequence classification (`BertForSequenceClassification`) is loaded from the `bert-base-uncased` checkpoint. The `num_labels` parameter is set to the number of unique request types in the dataset, which configures the classification head of the model with the appropriate output dimension.
    * **Code:** The code uses `BertForSequenceClassification.from_pretrained()` to load the model, passing the pre-trained checkpoint name and the number of labels.
    * **Reasoning:** Using a pre-trained model like BERT leverages the knowledge it has already acquired from a massive amount of text data. Fine-tuning this model on the specific email classification task requires less data and training time compared to training a model from scratch and often results in better performance.

7.  **✅ STEP 7: METRICS:**
    * **Mechanism:** The `accuracy` metric is loaded from the `evaluate` library. The `compute_metrics` function is defined to calculate this metric based on the model's predictions and the true labels during evaluation.
    * **Code:** `evaluate.load("accuracy")` loads the metric, and `compute_metrics` takes `eval_pred` (containing logits and labels), converts logits to predictions using `np.argmax()`, and then uses the loaded metric to compute the accuracy.
    * **Reasoning:** Evaluation metrics are essential for quantifying the performance of the trained model. Accuracy is a common and intuitive metric for classification tasks, representing the percentage of correctly classified instances.

8.  **✅ STEP 8: TRAINING ARGUMENTS:**
    * **Mechanism:** The `TrainingArguments` class from the Transformers library is used to configure various hyperparameters and settings for the training process. This includes the output directory for saving model checkpoints, the number of training epochs, batch sizes for training and evaluation, the evaluation and saving strategies, and settings for loading the best model at the end of training.
    * **Code:** An instance of `TrainingArguments` is created with several parameters set to control the training behavior.
    * **Reasoning:** Properly configured training arguments are crucial for effective model training. They control the duration of training, the amount of data processed in each step, when evaluation is performed, when and how model checkpoints are saved, and which metric to use for determining the best model.

9.  **✅ STEP 9: TRAIN:**
    * **Mechanism:** The `Trainer` class from the Transformers library is initialized with the loaded model, training arguments, training dataset, evaluation dataset, tokenizer, and the `compute_metrics` function. The `trainer.train()` method then starts the fine-tuning process of the BERT model on the training data, with evaluation performed at the specified intervals.
    * **Code:** An instance of `Trainer` is created, and its `train()` method is called.
    * **Reasoning:** The `Trainer` class provides a high-level API for training Transformers models, handling the training loop, evaluation, and checkpoint saving automatically based on the provided arguments.

10. **✅ STEP 10: SAVE MODEL:**
    * **Mechanism:** After the training process is complete, the fine-tuned BERT model and the tokenizer are saved to specified directories. This allows the trained model to be loaded and used for inference later without needing to retrain it.
    * **Code:** The `model.save_pretrained()` and `tokenizer.save_pretrained()` methods are used to save the model and tokenizer, respectively.
    * **Reasoning:** Saving the trained model is essential for deploying and using it for real-world classification tasks. It preserves the learned weights and the vocabulary mapping needed for processing new text data.

11. **✅ STEP 11: INFERENCE FUNCTION:**
    * **Mechanism:** The `predict_email_verbose()` function takes a text input (email content), tokenizes it using the saved tokenizer, and feeds it to the loaded BERT model in evaluation mode (disabling gradient calculations). The model outputs logits, which are then converted into probabilities using the softmax function. The function prints the probabilities for each request type and returns the predicted request type with its confidence score.
    * **Code:** The function loads the tokenizer and model (optionally moving the model to a GPU if available), tokenizes the input text, performs a forward pass through the model, applies softmax to the logits to get probabilities, and then identifies the class with the highest probability as the prediction.
    * **Reasoning:** This function provides a way to use the trained BERT model to classify new, unseen email or document text. The verbose output of probabilities can be helpful for understanding the model's confidence in its predictions.

12. **✅ STEP 12: ENSEMBLE CLASSIFIER (WITHOUT HF):**
    * **Mechanism:** The `ensemble_classify()` function combines the predictions of the fine-tuned BERT model (`predict_email_verbose()`) and Google Gemini (`classify_gemini()`) to make a final classification. It first gets the prediction and confidence from the BERT model. Then, it queries Gemini with a prompt asking it to classify the input text into one of the predefined `REQUEST_TYPES`. A voting mechanism is used to determine the majority prediction. Additionally, a confidence-based fallback is implemented: if the BERT model's confidence is below a threshold (0.5), the prediction from Gemini is preferred (if available). Finally, a keyword-based override (`priority_override()`) is applied to potentially adjust the final prediction based on the presence of specific keywords in the text.
    * **Code:** The `ensemble_classify()` function calls `predict_email_verbose()` and `classify_gemini()`. It then implements the voting logic, the confidence-based fallback, and calls `priority_override()`.
    * **Reasoning:** Ensemble methods often lead to improved performance and robustness compared to relying on a single model. Combining the strengths of a fine-tuned discriminative model (BERT) with a generative LLM (Gemini) can result in more accurate and reliable classifications. The confidence-based fallback helps to mitigate the risk of low-confidence predictions from the BERT model. The keyword-based override allows for incorporating domain-specific rules and improving accuracy for cases where simple keyword matching is highly indicative of a particular request type.

13. **✅ STEP 13: PDF TEXT EXTRACTION:**
    * **Mechanism:** The `extract_text_from_pdf()` function uses the `PyPDF2` library to extract text content from a given PDF file. It opens the PDF file in read-binary mode, creates a `PdfReader` object, and then iterates through each page of the PDF, extracting the text content and appending it to a single string.
    * **Code:** The function uses `PdfReader()` to open and read the PDF, iterates through `reader.pages`, and uses `page.extract_text()` to get the text from each page.
    * **Reasoning:** This functionality allows the system to process documents in PDF format, which is a common format for business communications. By extracting the text, the content of PDF documents can be fed into the classification pipeline in the same way as email text.

14. **✅ STEP 14: SYNTHETIC DATA GENERATION (Detailed Expansion):**
    * **Mechanism:** (Already detailed in Step 4, this section provides further context on the expanded generation). The synthetic data generation is expanded to cover all the main request types defined in `REQUEST_TYPES`. For each type, a set of templates designed to capture the essence of that request is used. The number of synthetic examples generated for each type is also increased to ensure better coverage in the training data.
    * **Code:** The code includes template lists and generation loops for "Adjustment", "AU Transfer", "Closing Notice", "Commitment Change", "Fee Payment", and "Money Movement - Outbound", following the same logic as the initial "Money Movement-Inbound" generation.
    * **Reasoning:** Expanding the synthetic data generation to all request types helps to create a more balanced and comprehensive training dataset. This is crucial for training a multi-class classification model that can accurately distinguish between all the defined categories. The templates are carefully crafted to represent the typical language and information content of each request type.

15. **✅ STEP 15: SUB-REQUEST CLASSIFICATION:**
    * **Mechanism:** The `classify_sub_request()` function takes the input text and the main predicted request type as arguments. It first checks if the main request type has any defined sub-request types in the `sub_request_mapping` (a filtered version of `REQUEST_TYPES` containing only main types with sub-types). If sub-types exist, it constructs a prompt for Google Gemini, instructing it to classify the input text into one of the sub-request options. The function then sends this prompt to the Gemini API and returns the classified sub-request type.
        * **Code:** The function retrieves the sub-request options for the given `main_class` from `sub_request_mapping`. It then creates a prompt string that includes the sub-request options and the input text. The `genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)` method is used to get the sub-classification from Gemini. A `try...except` block handles potential errors during the API call.
        * **Reasoning:** Sub-request classification provides a more detailed categorization of the incoming communication. By using a powerful LLM like Gemini, the system can leverage its understanding of language to identify nuanced differences between sub-requests within a broader category. This allows for more precise routing and handling of requests. The `sub_request_mapping` ensures that sub-classification is only attempted for main request types that have defined sub-types.

16. **✅ STEP 16: TEST CLASSIFICATION:**
    * **Mechanism:** This step demonstrates how to use the developed system for classifying a new document. It first extracts the text content from a sample PDF file (`sample1.pdf`) using the `extract_text_from_pdf()` function. Then, it uses the `ensemble_classify()` function to get the final classification result, which combines the predictions from the fine-tuned BERT model and Google Gemini, potentially applying keyword overrides. Finally, if a sub-request mapping exists for the predicted main request type, it calls the `classify_sub_request()` function to get a more granular classification.
    * **Code:** The code calls `extract_text_from_pdf()` with the path to the sample PDF, then calls `ensemble_classify()` with the extracted text. It prints the resulting classification dictionary. If a sub-request is applicable, it calls `classify_sub_request()` and prints the sub-request type.
    * **Reasoning:** This step serves as an end-to-end test of the classification pipeline, demonstrating how to process a document from text extraction to final (and potentially sub-) classification using the ensemble approach. It highlights the integration of the different components of the system.

17. **Conclusion**

This Smart Email and Document Triage System provides a robust and flexible solution for automating the classification of incoming communications. By combining the strengths of a fine-tuned BERT model for efficient pattern recognition and Google Gemini for its broad language understanding and reasoning capabilities, the system aims to achieve high accuracy and handle a wide range of request types. The inclusion of synthetic data generation, ensemble classification with confidence-based fallback and keyword overrides, and PDF text extraction further enhances the system's practicality and effectiveness in real-world applications. The modular design and detailed configuration allow for future expansion and customization to adapt to evolving business needs and new request types.
