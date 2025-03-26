from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class RequestType(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    description = db.Column(db.Text)
    keywords = db.Column(db.Text)

    def __repr__(self):
        return f"<RequestType {self.name}>"

    def as_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class ProcessedEmail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    filepath = db.Column(db.String(300), nullable=False)
    processing_date = db.Column(db.DateTime, default=datetime.utcnow)
    email_hash = db.Column(db.String(64), unique=True, nullable=False)
    primary_intent = db.Column(db.String(200))
    summary = db.Column(db.Text)
    request_types_json = db.Column(db.Text)
    extracted_data_json = db.Column(db.Text)

    def __repr__(self):
        return f"<ProcessedEmail {self.filename} - {self.processing_date}>"