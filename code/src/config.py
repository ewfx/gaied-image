import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'rgeovbettnvetkbn'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///site.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'uploads'
    UPLOAD_EXTENSIONS = {'eml', 'pdf', 'docx'}
    DEFAULT_REQUEST_TYPES = ['Information Request', 'Service Request', 'Problem Report', 'Change Request']