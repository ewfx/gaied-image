# app.py
import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import FileField, SubmitField, StringField, TextAreaField, HiddenField
from wtforms.validators import DataRequired
from config import Config
from werkzeug.utils import secure_filename
from models import db, RequestType, ProcessedEmail
import hashlib
from eml_parser import EmlParser
from datetime import datetime
import PyPDF2
from bs4 import BeautifulSoup
import json
from docx import Document
from dotenv import load_dotenv
from ai_integration import analyze_text  # Import the Gemini integration

load_dotenv()

app = Flask(__name__)
app.config.from_object(Config)
csrf = CSRFProtect(app)
db.init_app(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def initialize_default_request_types():
    with app.app_context():
        for rt_name in app.config['DEFAULT_REQUEST_TYPES']:
            if not RequestType.query.filter_by(name=rt_name).first():
                default_rt = RequestType(name=rt_name)
                db.session.add(default_rt)
        db.session.commit()

def create_tables():
    with app.app_context():
        db.create_all()

@app.before_request
def before_request_func():
    if not getattr(before_request_func, '_has_run', False):
        create_tables()
        initialize_default_request_types()
        before_request_func._has_run = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1].lower() in app.config['UPLOAD_EXTENSIONS']

def generate_email_hash(filepath):
    with open(filepath, 'rb') as f:
        content = f.read()
        return hashlib.sha256(content).hexdigest()

def parse_eml_content(filepath):
    parser = EmlParser(include_raw_body=True, include_attachment_data=False)
    with open(filepath, 'rb') as f:
        eml = parser.decode(f.read())
    subject = eml.get('subject')
    from_addr = eml.get('from')
    to_addr = eml.get('to')
    date_obj = eml.get('date')
    date_str = datetime.strftime(date_obj, '%Y-%m-%d %H:%M:%S') if date_obj else "N/A"
    body_parts = eml.get_body()
    body_text = ""
    if body_parts:
        plain_text_body = next((part[1] for part in body_parts if part[0] == 'text/plain'), None)
        if plain_text_body:
            body_text = plain_text_body
        else:
            html_body = next((part[1] for part in body_parts if part[0] == 'text/html'), None)
            if html_body:
                soup = BeautifulSoup(html_body, 'html.parser')
                body_text = soup.get_text(separator='\n')
    attachments = [att.get('filename') for att in eml.get_attachments()]
    return {'subject': subject, 'sender': from_addr, 'recipient': to_addr, 'date': date_str, 'body': body_text, 'attachments': attachments}

def extract_text_from_pdf(filepath):
    text = ""
    try:
        with open(filepath, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def extract_text_from_docx(filepath):
    text = ""
    try:
        document = Document(filepath)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return text

class UploadFileForm(FlaskForm):
    file = FileField("Select File", validators=[DataRequired()])
    submit = SubmitField("Upload")

class ConfigureRequestTypeForm(FlaskForm):
    id = HiddenField()
    name = StringField("Name", validators=[DataRequired()])
    description = TextAreaField("Description")
    keywords = StringField("Keywords (comma-separated)")
    submit = SubmitField("Save")

class AddRequestTypeForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    description = TextAreaField("Description")
    keywords = StringField("Keywords (comma-separated)")
    submit = SubmitField("Add Request Type")

class SaveFeedbackForm(FlaskForm):
    request_types_json = HiddenField()
    extracted_data_json = HiddenField()
    submit = SubmitField("Save Feedback")

# Define a form for the process_email page
class ProcessEmailForm(FlaskForm):
    submit = SubmitField("Start Processing")

@app.route('/', methods=['GET'])
def index():
    return redirect(url_for('upload_file_page'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_file_page():
    form = UploadFileForm()
    if form.validate_on_submit():
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        file = form.file.data
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
                session['uploaded_filepath'] = filepath
                session['uploaded_filename'] = filename
                return redirect(url_for('process_email_page'))
            except Exception as e:
                flash(f'Error saving file: {e}', 'error')
                return redirect(request.url)
        else:
            flash(f'Invalid file type. Allowed types: {", ".join(app.config["UPLOAD_EXTENSIONS"])}', 'error')
            return redirect(request.url)
    return render_template('upload.html', form=form, error=request.args.get('error'))

@app.route('/configure', methods=['GET', 'POST'])
def configure_request_types():
    with app.app_context():
        session['use_default_presets'] = session.get('use_default_presets', True)
        custom_request_types_from_db = [request_type.as_dict() for request_type in RequestType.query.filter(RequestType.name.notin_(app.config['DEFAULT_REQUEST_TYPES'])).all()]
        default_request_types = app.config['DEFAULT_REQUEST_TYPES']

        add_form = AddRequestTypeForm(prefix="add")
        edit_form = ConfigureRequestTypeForm(prefix="edit")

        if add_form.add_request_type.data and add_form.validate_on_submit():
            name = add_form.name.data.strip()
            description = add_form.description.data.strip()
            keywords_str = add_form.keywords.data.strip()
            if not RequestType.query.filter_by(name=name).first():
                new_rt = RequestType(name=name, description=description, keywords=keywords_str)
                db.session.add(new_rt)
                db.session.commit()
                return jsonify({'success': True, 'message': f"Request type '{name}' added successfully.", 'request_type': new_rt.as_dict()})
            else:
                return jsonify({'success': False, 'error': f"Request type '{name}' already exists."})

        if edit_form.save.data and edit_form.validate_on_submit():
            rt_id = edit_form.id.data
            rt_to_update = RequestType.query.get(rt_id)
            if rt_to_update:
                rt_to_update.name = edit_form.name.data.strip()
                rt_to_update.description = edit_form.description.data.strip()
                rt_to_update.keywords = edit_form.keywords.data.strip()
                db.session.commit()
                return jsonify({'success': True, 'message': f"Request type '{rt_to_update.name}' updated successfully.", 'request_type': rt_to_update.as_dict()})
            else:
                return jsonify({'success': False, 'error': 'Request type not found for editing.'})

        return render_template('configure.html', use_default_presets=session.get('use_default_presets', True),
                               default_request_types=default_request_types,
                               custom_request_types=custom_request_types_from_db,
                               add_form=add_form, edit_form=edit_form)

@app.route('/configure/edit/<int:id>')
def get_request_type_for_edit(id):
    with app.app_context():
        request_type = RequestType.query.get(id)
        if request_type:
            return jsonify({'request_type': request_type.as_dict()})
        else:
            return jsonify({'error': 'Request type not found.'})

@app.route('/configure/delete/<int:id>', methods=['POST'])
def delete_request_type(id):
    with app.app_context():
        request_type = RequestType.query.get(id)
        if request_type:
            db.session.delete(request_type)
            db.session.commit()
            return jsonify({'success': True, 'message': f"Request type '{request_type.name}' deleted successfully."})
        else:
            return jsonify({'error': 'Request type not found.'})

@app.route('/configure/toggle_presets', methods=['POST'])
def toggle_presets():
    data = request.get_json()
    if data is not None and 'use_presets' in data:
        session['use_default_presets'] = data['use_presets']
        return jsonify({'success': True})
    return jsonify({'error': 'Invalid request.'}), 400

@app.route('/process_email', methods=['GET', 'POST'])
def process_email_page():
    if 'uploaded_filepath' not in session:
        return redirect(url_for('upload_file_page'))

    filepath = session['uploaded_filepath']
    filename = session['uploaded_filename']
    file_extension = filename.rsplit('.', 1)[-1].lower()

    # Instantiate the form
    form = ProcessEmailForm()

    with app.app_context():
        custom_request_types = RequestType.query.filter(
            RequestType.name.notin_(app.config['DEFAULT_REQUEST_TYPES'])
        ).all()
        custom_request_type_keywords = {
            rt.name: [keyword.strip() for keyword in rt.keywords.split(',') if keyword.strip()]
            for rt in custom_request_types
        }

        preview_data = {'subject': filename, 'attachments': [], 'body_snippet': 'Preview not available.'}

        if file_extension == 'eml':
            preview_data = parse_eml_content(filepath)
            preview_data['body_snippet'] = preview_data.get('body', '')[:200] + "..." if preview_data.get('body') else "No body."
        elif file_extension == 'pdf':
            preview_data['body_snippet'] = extract_text_from_pdf(filepath)[:200] + "..." if extract_text_from_pdf(filepath) else "No text extracted."
        elif file_extension == 'docx':
            preview_data['body_snippet'] = extract_text_from_docx(filepath)[:200] + "..." if extract_text_from_docx(filepath) else "No text extracted."
        # Handle other file types for preview if needed

        if form.validate_on_submit():
            email_hash = generate_email_hash(filepath)
            existing_email = ProcessedEmail.query.filter_by(email_hash=email_hash).first()

            if file_extension == 'eml':
                preview_data = parse_eml_content(filepath)
            elif file_extension == 'pdf':
                preview_data['body_snippet'] = extract_text_from_pdf(filepath)[:200] + "..." if extract_text_from_pdf(filepath) else "No text extracted."
            elif file_extension == 'docx':
                preview_data['body_snippet'] = extract_text_from_docx(filepath)[:200] + "..." if extract_text_from_docx(filepath) else "No text extracted."

            if existing_email:
                return render_template('processing_output.html', upload_successful=True, is_duplicate=True,
                                       email_preview=preview_data, attachments=preview_data.get('attachments', []),
                                       extracted_data={}, ai_analysis={})
            else:
                extracted_text = ""
                if file_extension == 'eml':
                    extracted_text = preview_data.get('body', '')
                elif file_extension == 'pdf':
                    extracted_text = extract_text_from_pdf(filepath)
                elif file_extension == 'docx':
                    extracted_text = extract_text_from_docx(filepath)
                # Handle other file types similarly

                try:
                    ai_analysis_results = analyze_text(extracted_text)
                except Exception as e:
                    print(f"Error during AI analysis: {e}")
                    ai_analysis_results = {'error': f'Error during AI analysis: {e}'}

                if 'request_types' in ai_analysis_results and isinstance(ai_analysis_results['request_types'], list):
                    enhanced_request_types = []
                    for ai_rt in ai_analysis_results['request_types']:
                        matched_custom_rt = None
                        if isinstance(ai_rt, dict) and 'request_type' in ai_rt:
                            for custom_name, custom_keywords in custom_request_type_keywords.items():
                                if custom_name.lower() in ai_rt['request_type'].lower() or \
                                   any(keyword.lower() in extracted_text.lower() for keyword in custom_keywords):
                                    matched_custom_rt = custom_name
                                    break
                            if matched_custom_rt:
                                enhanced_request_types.append({
                                    'request_type': matched_custom_rt,
                                    'confidence': ai_rt.get('confidence', None),
                                    'sub_classifications': ai_rt.get('sub_classifications', []),
                                    'from_custom': True
                                })
                            else:
                                enhanced_request_types.append(ai_rt)
                    ai_analysis_results['request_types'] = enhanced_request_types
                elif 'request_types' not in ai_analysis_results:
                    ai_analysis_results['request_types'] = []

                new_processed_email = ProcessedEmail(
                    filename=filename,
                    filepath=filepath,
                    email_hash=email_hash,
                    primary_intent=ai_analysis_results.get('primary_intent'),
                    summary=ai_analysis_results.get('summary'),
                    request_types_json=json.dumps(ai_analysis_results.get('request_types', [])),
                    extracted_data_json=json.dumps(ai_analysis_results.get('extracted_data', {}))
                )
                db.session.add(new_processed_email)
                db.session.commit()

                return render_template('processing_output.html', upload_successful=True, is_duplicate=False,
                                       email_preview=preview_data, attachments=preview_data.get('attachments', []),
                                       extracted_data=ai_analysis_results.get('extracted_data', {}),
                                       ai_analysis=ai_analysis_results,
                                       processed_email_id=new_processed_email.id)

    return render_template('process_email.html', filename=filename, email_preview=preview_data, form=form)

@app.route('/save_feedback/<int:id>', methods=['POST'])
def save_feedback(id):
    with app.app_context():
        processed_email = ProcessedEmail.query.get_or_404(id)
        form = SaveFeedbackForm()
        if form.validate_on_submit():
            edited_request_types = json.loads(form.request_types_json.data)
            edited_extracted_data = json.loads(form.extracted_data_json.data)

            processed_email.request_types_json = json.dumps(edited_request_types)
            processed_email.extracted_data_json = json.dumps(edited_extracted_data)
            db.session.commit()
            flash('Feedback saved successfully.', 'success')
            return redirect(url_for('processing_history'))
        else:
            flash('Error saving feedback.', 'error')
            return redirect(url_for('processing_history'))

@app.route('/history')
def processing_history():
    with app.app_context():
        processed_emails = ProcessedEmail.query.order_by(ProcessedEmail.processing_date.desc()).all()
        history_data = []
        for email in processed_emails:
            history_data.append({
                'id': email.id,
                'filename': email.filename,
                'processing_date': email.processing_date,
                'primary_intent': email.primary_intent,
                'summary': email.summary,
                'request_types': json.loads(email.request_types_json) if email.request_types_json else [],
                'extracted_data': json.loads(email.extracted_data_json) if email.extracted_data_json else {}
            })
        return render_template('history.html', history=history_data)

@app.route('/history/view/<int:id>')
def view_processed_item(id):
    with app.app_context():
        processed_email = ProcessedEmail.query.get_or_404(id)
        email_preview_data = {}
        try:
            if os.path.exists(processed_email.filepath):
                file_extension = processed_email.filename.rsplit('.', 1)[-1].lower()
                if file_extension == 'eml':
                    email_preview_data = parse_eml_content(processed_email.filepath)
                elif file_extension == 'pdf':
                    email_preview_data['subject'] = processed_email.filename
                    email_preview_data['attachments'] = [processed_email.filename]
                    email_preview_data['body_snippet'] = extract_text_from_pdf(processed_email.filepath)[:200] + "..."
                elif file_extension == 'docx':
                    email_preview_data['subject'] = processed_email.filename
                    email_preview_data['attachments'] = [processed_email.filename]
                    email_preview_data['body_snippet'] = extract_text_from_docx(processed_email.filepath)[:200] + "..."
        except Exception as e:
            print(f"Error generating preview for {processed_email.filename}: {e}")

        feedback_form = SaveFeedbackForm(
            request_types_json=processed_email.request_types_json,
            extracted_data_json=processed_email.extracted_data_json
        )

        return render_template('view_processed.html',
                               processed_item=processed_email,
                               request_types=json.loads(processed_email.request_types_json) if processed_email.request_types_json else [],
                               extracted_data=json.loads(processed_email.extracted_data_json) if processed_email.extracted_data_json else {},
                               email_preview=email_preview_data,
                               feedback_form=feedback_form)

@app.route('/reset_presets', methods=['POST'])
def reset_presets():
    with app.app_context():
        RequestType.query.filter(RequestType.name.notin_(app.config['DEFAULT_REQUEST_TYPES'])).delete()
        db.session.commit()
        return redirect(url_for('configure_request_types'))

if __name__ == '__main__':
    app.run(debug=True)