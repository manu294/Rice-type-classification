# forms/__init__.py

# This file makes 'forms' a Python package
# You can define WTForms here or import them from other files

# Example form setup (optional, only if using Flask-WTF)

from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired

class UploadForm(FlaskForm):
    image = FileField('Upload Rice Image', validators=[DataRequired()])
    submit = SubmitField('Predict')
