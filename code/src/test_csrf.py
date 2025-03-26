from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import FileField, SubmitField
from os import urandom

app = Flask(__name__)
app.config['SECRET_KEY'] = urandom(24)

csrf = CSRFProtect(app)  # Initialize CSRF

# Create a Flask-WTF form
class UploadForm(FlaskForm):
    file = FileField("Upload File")
    submit = SubmitField("Upload")

@app.route('/upload_test', methods=['GET', 'POST'])
def upload_test():
    form = UploadForm()  # Create form instance
    if request.method == 'POST' and form.validate_on_submit():
        return "File uploaded successfully (CSRF validated)"
    return render_template('upload_test.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
