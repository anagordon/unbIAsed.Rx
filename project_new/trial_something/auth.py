import os
import re
import json
from flask import Blueprint, jsonify, render_template, request, flash, redirect, session, url_for
import pandas as pd
from .models import User, Drugs
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   #means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Email
from werkzeug.utils import secure_filename
from fuzzywuzzy import process
import pytesseract
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import difflib
from sqlalchemy import or_
import torch
import requests
import torch.nn as nn
import torch
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
import timm
from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    Rotate, ShiftScaleRotate, Transpose
)
from albumentations.pytorch import ToTensorV2
# from trial_something.views import get_model
from .utils import get_model
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import openai
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer, AutoModel
import seaborn as sns
import tensorflow_hub as hub
import faiss
import numpy as np

# Set environment variables to resolve OpenMP runtime conflict and disable oneDNN custom operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["OPENAI_API_KEY"] = "" #make sure to insert your own API key here


class BaselineModel(nn.Module):
    def __init__(self, num_classes: int, model_type = 'shuffleNet'):
        super().__init__()
        self.model_type = model_type
        if model_type == 'shuffleNet':
          self.model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
          self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_type == 'ResNext':
          self.model = timm.create_model('resnext50_32x4d', pretrained=True)
          n_features = self.model.fc.in_features
          self.model.fc = nn.Linear(n_features, num_classes)
        elif model_type == 'resnet50':
          backbone = timm.create_model(model_type, pretrained=True)
          n_features = backbone.fc.in_features
          self.backbone = nn.Sequential(*backbone.children())[:-2]
          self.classifier = nn.Linear(n_features, num_classes)
          self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
      if self.model_type != 'resnet50':
        return self.model(x)
      elif self.model_type == 'resnet50':
        x = self.backbone(x)
        feats = x
        x = self.pool(feats).view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def load(filename: str):
        checkpoint = torch.load(filename)
        model = BaselineModel(checkpoint["num_classes"])
        model.load_state_dict(checkpoint["params"])
        return model

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if email == '' or password == '':
            flash('Please fill in all fields.', category='error')
        elif user:
            if check_password_hash(user.password, password):
                login_user(user, remember=True)
                session['device_seen'] = True
                return redirect(url_for('views.home'))
            
            elif not check_password_hash(user.password, password):
                flash('Incorrect password, try again.', category='error')

                return redirect(url_for('auth.login'))
            else:
                flash('Email does not exist.', category='error')
                return redirect(url_for('auth.login'))
        else:
            flash('Email does not exist.', category='error')

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("login-mobile.html", user=current_user)
    return render_template("login.html", user=current_user)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', category='success')
    return redirect(url_for('views.opening'))

class UpdateUserForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(message = 'Enter a valid email'),Email()])
    firstname = StringField('First Name', validators=[DataRequired(message = 'Enter your first name')])
    
    submit = SubmitField('Update!')

    def validate_email(self, email):
        if User.query.filter_by(email=email.data).first():
            raise ValidationError('This email has been registered already!')


@auth.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    first_name = request.form.get('first_name')
    last_name = request.form.get('last_name')
    sexe = request.form.get('sexe')

    current_user.first_name = first_name
    current_user.last_name = last_name
    current_user.sexe = sexe
    db.session.commit()

    return redirect(url_for('auth.profile'))  # Redirect to the profile page


@auth.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("profile-mobile.html", user=current_user)
    return render_template("profile.html", user=current_user)

@auth.route('/saved')
@login_required
def saved():
    # drugs = Drugs.query.all()

    # for drug in drugs:
    #     drug.is_saved = drug in current_user.drugs
    drugs = current_user.drugs

    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("saved-mobile.html", drugs=drugs, user=current_user)
    return render_template("saved.html", drugs=drugs, user=current_user)
 
@auth.route('/about')
def about():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('about-mobile.html', user=current_user)
    return render_template('about.html', user=current_user)

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName').capitalize()
        last_name = request.form.get('lastName').capitalize()
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        sex = request.form.get('sexSelect', '').strip()

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already exists.', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')
        elif len(first_name) < 2:
            flash('First name must be greater than 1 character.', category='error')
        elif password1 != password2:
            flash('Passwords don\'t match.', category='error')
        elif len(password1) < 7:
            flash('Password must be at least 7 characters.', category='error')
        else:
            new_user = User(email=email, sexe = None, first_name=first_name, last_name=last_name, password=generate_password_hash(
                password1, method='pbkdf2:sha256'))
            
            if sex == 'female' or sex == 'male':
                new_user.sexe = sex.capitalize()
            elif sex =='none':
                new_user.sexe = None
            
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            session['device_seen'] = True
            flash('Account created!', category='success')
            return redirect(url_for('views.home'))
            
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("sign_up-mobile.html", user=current_user)
    return render_template("sign_up.html", user=current_user)

@auth.route('/delete_account', methods=['GET','POST'])
@login_required
def delete_account():
    #go look at Modal (bootstrap) for this
    try:
        user = User.query.filter_by(id=current_user.id).first()

        # Delete the user
        db.session.delete(user)

        # Commit the changes to the database
        db.session.commit()

        # Log out the user
        logout_user()

        flash("Your account has been deleted successfully.")
        return redirect(url_for('views.opening'))

    except Exception as e:
        print("Error deleting user account: ", e)
        flash("An error occurred while trying to delete your account.")
        return redirect(url_for('auth.profile'))

@auth.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        search_term = request.form.get('query')
    else:
        search_term = request.args.get('query')

    if search_term == "":
        flash("Please enter a search term.", category='error')
        return redirect(url_for('views.home'))

    # Perform a case-insensitive pattern search
    search_term = str(search_term) if search_term is not None else ''
    search_term = '%' + search_term + '%'
    # results = Drugs.query.filter(Drugs.name.ilike(search_term)).all()
    results = Drugs.query.filter(or_(Drugs.name.ilike(search_term), Drugs.disease.ilike(search_term))).all()

    if not results:
    # If no exact match is found, find the closest match
        all_drugs = Drugs.query.all()
        all_drug_names = [drug.name for drug in all_drugs]
        closest_match_result = process.extractOne(search_term, all_drug_names)
        if closest_match_result:  # Check if closest_match_result is not None
            closest_match_name = closest_match_result[0]
            closest_match = Drugs.query.filter_by(name=closest_match_name).first()
            if closest_match:
                results = [closest_match]
        else:
            flash("No drug found with that name.", category='error')
            return redirect(url_for('views.home'))
        
    disease_prevalence = None
    prediction_risk = ""
    flash_message_risk = ""
    errorFlash = False
    # prediction_risk_male = ""
    result_string_pred = ""
    
    drug_search = ""
    disease_search = "" 

    drug_search = request.form.get('drugName')
    disease_search = request.form.get('drugCondition')
    drug_id = request.form.get('drug_id')

    # drugs = Drugs.query.all()
    drug_id = request.form.get('drug_id')
    # drug = Drugs.query.get(drug_id)

    # if not drug_id or not drug_search or not disease_search:
    #     return redirect(url_for('views.home'))

    # if disease_search == None or drug_search == None:
    #     errorFlash = True
    #     # flash('Please fill in all fields', 'error')
    #     flash_message_risk = 'Please fill in all fields'
    #     user_agent = request.headers.get('User-Agent').lower()
    #     if 'mobile' in user_agent:
    #         return render_template("search_results-mobile.html", flash_message_risk=flash_message_risk, user=current_user, errorFlash=errorFlash)    
    #     return render_template("search_results.html", flash_message_risk=flash_message_risk, user=current_user,errorFlash=errorFlash)

    if drug_search:
        drug_search = drug_search.upper()
    if disease_search:
        disease_search = disease_search.upper()

    if (drug_search or disease_search):
        prediction_risk = get_model(drug_search, disease_search)

    # Convert drug_id to integer if it is not None
    if drug_id is not None:
        try:
            drug_id = int(drug_id)
        except ValueError:
            flash("Invalid drug ID.", category='error')
            # return redirect(url_for('views.home'))

    F = 0
    M = 0

    if prediction_risk is not (None or ""):
        R = 0
        M = 100 - prediction_risk
        F = prediction_risk

        if current_user.is_authenticated and current_user.sexe is not None:
            
            if current_user.sexe.lower() == 'male':

                if M > F:
                    R = M - F
                    R = str(round(R,2))
                    M = str(round(M,2))
                    result_string_pred = (
        f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
        f"Male patients have a <b>{R}%</b> lower risk of developing a reaction compared to female patients. <br>"
    )
                else:
                    R = F - M
                    R = str(round(R,2))
                    M = str(round(M,2))
                    result_string_pred = (
        f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
        f"Male patients have an additional <b>{R}%</b> risk of developing a reaction compared to female patients.<br> "
    
    )
            elif current_user.sexe.lower() == 'female':
                if F > M:
                    R = F - M
                    R = str(round(R,2))
                    F = str(round(F,2))
                    result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients. <br>"
        
    )
                else:
                    R = M - F
                    R = str(round(R,2))
                    F = str(round(F,2))
                    result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients. <br>"
    )
        else:
            if F > M:
                R = F - M
                R = str(round(R,2))
                F = str(round(F,2))
                result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients.<br>"
    )
            else:
                R = M - F
                R = str(round(R,2))
                F = str(round(F,2))
                result_string_pred = (
        f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
        f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients.<br>"
    )
    result_string_pred = result_string_pred  
    
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template("search_results-mobile.html", search_term=search_term, results=results, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)
    return render_template("search_results.html",  search_term=search_term, results=results, user=current_user, disease_prevalence=disease_prevalence, result_string_pred=result_string_pred, result_drug_id=drug_id)

def preprocess(sentence):
    # Remove punctuation
    sentence = re.sub(r'[^\w\s]', '', sentence)
    # Convert to uppercase
    sentence = sentence.lower()
    # Tokenize
    words = sentence.split()
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    # Join words back into sentence
    sentence = ' '.join(words)
    return sentence

def predict_with_cnn(image_path, model_path, input_width, input_height):
    # Define the transformation chain
    transform = Compose([
        RandomResizedCrop(input_height, input_width),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    # print('Processing image')
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load image and preprocess
    image = cv2.imread(image_path)
    augmented = transform(image=image)
    image = augmented['image']  # Apply transformations
    image = image.unsqueeze(0)  # Add batch dimension
    # print('Done processing image')

    # Load the model
    model = BaselineModel(10, 'resnet50')
    # model = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    # Make prediction
    with torch.no_grad():
        output = model(image.to(device))

    # Interpret results as classification
    probs = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probs, dim=1).item()
    return predicted_class


# Function to create a requests session with retries
def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


@auth.route('/identify', methods=['GET','POST'])
def identify():
    sentence = ""
    text=""
    word=None
    something=""
    prediction_risk = ""
    # prediction_risk_male = ""
    result_string = ""
    errorFlash = False
    flash_message_label = ""
    flash_message_pill = ""
    flash_message_risk = ""
    # result = ""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
    csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
    df = pd.read_csv(csv_file)
    df.iloc[1:,0] = df.iloc[1:,0].str.lower()
    meds = df.iloc[1:,0].tolist()
    
    
    if request.method == 'POST':
        # #variables for risk
        # Medication = request.form.get('Medication')
        # Age = request.form.get('Age')
        # Sex = request.form.get('Sex')
        # Disease = request.form.get('Disease')
        # Ethnicity = request.form.get('Ethnicity')

        # # Initialize the results dictionary
        # results_dict = {
        #     "Clinical_trial_data": "",
        #     "Relevant_ADR_reports": "",
        #     "ADR_statistics": ""
        # }

        image_file = request.files.get('uploaded-pill-image')
        label_file = request.files.get('uploaded-label-image')
        button_clicked1 = request.form.get('submit-button1')
        button_clicked2 = request.form.get('submit-button2')
        drug_search = ""
        drug_menu = request.form.get('drugRiskSelect')

        if drug_menu == 'input':
            drug_search = request.form.get('drugRiskInput')
        elif drug_menu == 'select':
            drug_search = request.form.get('selectDrug')
        else:
            flash('Please select a valid option', 'error')
            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", user=current_user,meds=meds)
            return render_template("identify.html", user=current_user,meds=meds)

        disease_search = request.form.get('diseaseRisk')
        button_clicked3 = request.form.get('submit-button3')
        button_clicked2 = request.form.get('submit-button2')

        # Check if at least one file was uploaded
        # if image_file is None and label_file is None: #and drug_search == '' and disease_search == ''
        #     errorFlash = True
        #     # flash('Either an image or a label file must be uploaded', 'error')
        #     flash_message = 'Either an image or a label file must be uploaded'
        #     flash_message_label = flash_message
        #     flash_message_pill = flash_message  

        #     user_agent = request.headers.get('User-Agent').lower()
        #     if 'mobile' in user_agent:
        #         return render_template("identify-mobile.html", flash_message_pill = flash_message_pill,  flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)  
        #     return render_template("identify.html", flash_message_pill = flash_message_pill, flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)

        if image_file is None and (drug_search is None and disease_search is None): #pill
            errorFlash = True
            flash_message_pill = 'An image file must be uploaded'

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", flash_message_pill = flash_message_pill, user=current_user,meds=meds, errorFlash=errorFlash)  
            return render_template("identify.html", flash_message_pill = flash_message_pill, user=current_user,meds=meds, errorFlash=errorFlash)

        if label_file is None and (drug_search is None and disease_search is None): #label
            errorFlash = True
            flash_message_label = 'A label file must be uploaded'

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)
            return render_template("identify.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)

        # Process the image file if the 'image' button was clicked and a file was uploaded
        if button_clicked1 == 'label' and label_file and label_file.filename != '':
            label_filename = secure_filename(label_file.filename)
            label_filepath = os.path.join('/tmp', label_filename)
            label_file.save(label_filepath)
            
            # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            img = cv2.imread(label_filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes = pytesseract.image_to_data(img)

            for i, box in enumerate(boxes.splitlines()):
                if i==0:
                    continue
                box = box.split()
                if len(box) == 12:
                    sentence += box[11] + " "

            session["sentence"] = sentence

            sentence = preprocess(sentence)
            
            #read csv
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current file
            csv_file = os.path.join(BASE_DIR, 'final_results.csv')  # Join the base directory with the file name
            df = pd.read_csv(csv_file)
            df.iloc[1:,0] = df.iloc[1:,0].str.lower()
            meds = df.iloc[1:,0].tolist()

            for word in sentence.split():
                close_matches = difflib.get_close_matches(word, meds, n=1, cutoff=0.9)
                if close_matches:
                    # flash('Label successfully identified', 'success')
                    text = close_matches[0].capitalize() + ' has been found!'
                    something = close_matches[0]
                    break
                    
                else:
                    text = 'This drug is not currently in our database.'

            if sentence == "":
                errorFlash = True
                # flash('No text was found in the image', 'error')
                flash_message_label = 'No text was found in the image'

                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)
                return render_template("identify.html", flash_message_label = flash_message_label, user=current_user,meds=meds, errorFlash=errorFlash)

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", user=current_user, text=text, word=word, something=something,meds=meds, errorFlash=errorFlash) 
            return render_template("identify.html", user=current_user, text=text, word=word, something=something, meds=meds, errorFlash=errorFlash)
            # os.remove(label_filepath)


        # Process the label file if the 'label' button was clicked and a file was uploaded
        # elif button_clicked2 == 'pill' and image_file and image_file.filename != '':
            image_filename = secure_filename(image_file.filename)
            image_filepath = os.path.join('/tmp', image_filename)
            image_file.save(image_filepath)

            # Define the class dictionary
            class_dict = {'Alaxan': 0, 'Bactidol': 1, 'Biogesic': 2, 'Lamictal': 4, 'DayZinc': 3, 'Rivaroxaban': 5,
                'Fish Oil': 6, 'Kremil S': 7, 'Medicol': 8, 'Neozep': 9}
            
            # Create a reverse dictionary
            reverse_class_dict = {v: k for k, v in class_dict.items()}

            # Example usage
            #model_path = r'C:\Users\anbgo\coding_projects_flask\GitHub-Rx\project_new\trial_something\resnet50-2.pt'
            model_path = os.path.join(BASE_DIR, 'resnet50-2.pt')
            input_width = 224  # Replace with your model's input width
            input_height = 224  # Replace with your model's input height
            predicted_class = predict_with_cnn(image_filepath, model_path, input_width, input_height)
            predicted_class_name = reverse_class_dict[predicted_class]
            # print(f"Predicted class: {predicted_class_name}")
            pill = predicted_class_name + " has been found!"

            # flash('Pill successfully identified', 'success')
            # os.remove(image_filepath)

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", user=current_user, text=text, word=word, something=something, pill=pill,meds=meds, errorFlash=errorFlash) 
            return render_template("identify.html", user=current_user, text=text, word=word, something=something, pill=pill,meds=meds, errorFlash=errorFlash)
        
        elif button_clicked2 == 'risk-btn':
            #new code starts here
            #variables for risk
            Medication = request.form.get('Medication')
            Age = request.form.get('Age')
            Sex = request.form.get('Sex')
            Disease = request.form.get('Disease')
            Ethnicity = request.form.get('Ethnicity')

            # Initialize the results dictionary
            results_dict = {
                "Clinical_trial_data": "",
                "Relevant_ADR_reports": "",
                "ADR_statistics": ""
            }
            #EXTRACT THE CLINICAL TRIALS FOR A SPECIFIC MEDICATION AND DISEASE
            payload = {
                "query.cond": Disease,
                "query.term": Medication,
                "filter.overallStatus": 'COMPLETED',
                "query.intr": Medication,
                "sort": "@relevance:desc",

            }
            url = "https://clinicaltrials.gov/api/v2/studies"

            # Check if the request was successful
            response = requests.get(url, params=payload)

            list(response.json().items())[0]
            json.dumps(response.json()['studies'][0])

            class DotDict(dict):
                __getattr__ = dict.__getitem__
                __setattr__ = dict.__setitem__
                __delattr__ = dict.__delitem__

                def __init__(self, dct):
                    for key, value in dct.items():
                        if hasattr(value, 'keys'):
                            value = DotDict(value)
                        self[key] = value

            # Initialize an empty dictionary to store the results
            results_dict = {}
            trial_ids = []

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    data = response.json()
                    studies = data.get('studies', [])

                    # Collect clinical trial data as a string
                    clinical_trial_data = []
                    for idx, study_data in enumerate(studies):
                        study = DotDict(study_data)
                        try:
                            # Check if the required nested attributes are present
                            if (
                                hasattr(study, 'resultsSection') and
                                hasattr(study.resultsSection, 'baselineCharacteristicsModule') and
                                hasattr(study.resultsSection.baselineCharacteristicsModule, 'measures')
                            ):
                                clinical_trial_data.append(f"Study {idx + 1}:\n")
                                clinical_trial_data.append(f"{study.protocolSection.eligibilityModule}\n")
                                clinical_trial_data.append(f"{study.resultsSection.baselineCharacteristicsModule.measures}\n\n")
                        except (KeyError, AttributeError):
                            continue
                    results_dict["Clinical_trial_data"] = ''.join(clinical_trial_data)  # Save clinical trial data as string

                except requests.exceptions.JSONDecodeError:
                    print("Failed to decode JSON. Here is the raw response:")
                    # print(response.text)
            else:
                print(f"Request failed with status code: {response.status_code}")
                # print(response.text)

            #GET THE ADR REPORT STATISTICS FOR SPECIFIC MEDICATION
            csv_file_path = os.path.join(os.path.dirname(__file__), 'ADRdata.csv')
            df = pd.read_csv(csv_file_path, delimiter='\t')

            # Define the specific string you're looking for
            specific_string = Medication.upper()
            # Filter rows where the DRUGNAME column contains the specific string
            filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]

            # ---- GET AGE INFORMATION -----
            # Define age bins and labels for grouping
            age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
            age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

            # Categorize ages into bins
            df['AGE_GROUP'] = pd.cut(df['AGE_Y'], bins=age_bins, labels=age_labels, right=False)

            # Count occurrences for each age group and calculate percentages
            age_group_counts = df['AGE_GROUP'].value_counts().sort_index()
            age_group_percentages = (age_group_counts / age_group_counts.sum()) * 100

            # Select the top 3 age groups by count
            top_3_age_groups = age_group_percentages.nlargest(3).index.astype(str).tolist()
            age_group_sentence = f"The patients with the highest rate of reported drug reaction for this medicine are \"{top_3_age_groups[0]}\" then \"{top_3_age_groups[1]}\" then \"{top_3_age_groups[2]}\"."

            # ----- GET SEX INFORMATION ------
            gender_counts = df['GENDER_ENG'].value_counts()
            gender_percentages = (gender_counts / gender_counts.sum()) * 100

            female_percentage = gender_percentages.get('Female', 0)
            male_percentage = gender_percentages.get('Male', 0)
            gender_sentence = f"Out of all the reports, {female_percentage:.2f}% were women's and {male_percentage:.2f}% were men's."

            # ----- GET SERIOUSNESS INFORMATION -----
            seriousness_counts = df['SERIOUSNESS_ENG'].value_counts()
            seriousness_percentages = (seriousness_counts / seriousness_counts.sum()) * 100

            serious_percentage = seriousness_percentages.get('Serious', 0)
            not_serious_percentage = seriousness_percentages.get('Not Serious', 0)
            seriousness_sentence = f"Out of all the reported ADRs, {serious_percentage:.2f}% were serious and {not_serious_percentage:.2f}% were not serious."

            # Combine all sentences into one string and save to ADR_statistics key
            results_dict["ADR_statistics"] = f"{age_group_sentence}\n{gender_sentence}\n{seriousness_sentence}"

            #GET MOST RELEVANT ADR REPORTS
            #Load BERT model and tokenizer
            # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)

            model_str = "NeuML/pubmedbert-base-embeddings"
            tokenizer = AutoTokenizer.from_pretrained(model_str)
            model = AutoModel.from_pretrained(model_str)

            # Function to get sentence embeddings
            def get_sentence_embedding(sentences):
                batch_size = 32
                embeddings = []
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i + batch_size]  # Get a batch of sentences

                    encoded_inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                    with torch.no_grad():
                        outputs = model(**encoded_inputs)[0] # get the last hidden state vector
                    attention_mask = encoded_inputs['attention_mask'][..., None] # handle padding with attention mask

                    sentence_embedding = torch.sum(outputs * attention_mask, dim=1) / torch.clamp(torch.sum(attention_mask, dim=1), min=1e-9)
                    norms = torch.clamp(torch.sqrt(torch.sum(sentence_embedding**2, dim=1)), min=1e-9)[..., None]
                    sentence_embedding_normalized = sentence_embedding / norms

                    # Append embeddings for the batch to the overall list
                    embeddings.extend(sentence_embedding_normalized.cpu().numpy().astype('float32'))

                return embeddings

                    
            # Load FAISS index and metadata
            index_file_path = os.path.join(os.path.dirname(__file__), 'merged_index_19000.index')
            index = faiss.read_index(index_file_path)

            metadata_file_path = os.path.join(os.path.dirname(__file__), 'merged_metadata_19000.csv')
            metadata_df = pd.read_csv(metadata_file_path)

            # Function to query the FAISS database - UPDATED VIN'S VERSION november 21st
            def query_database(query_sentence, index_path, metadata_path, top_k=10):
                # Load the metadata
                if not os.path.exists(metadata_path):
                    raise Exception(f"Metadata file not found: {metadata_path}")
                metadata_df = pd.read_csv(metadata_path)

                # Generate query embedding
                query_embedding = np.array(get_sentence_embedding([query_sentence])).astype('float32')

                # Load the FAISS index
                if os.path.exists(index_path):
                    index = faiss.read_index(index_path)
                else:
                    raise Exception(f"No FAISS index file found at: {index_path}")

                # Perform the search
                similarities, indices = index.search(query_embedding, top_k * 3)
                distances = 1 - similarities  # Convert similarities to cosine distances

                # Filter results
                similar_sentences = []
                count = 0
                is_female_query = 'female' in query_sentence.lower()

                for i in range(len(indices[0])):
                    sentence = metadata_df.iloc[indices[0][i]]['sentence']
                    if is_female_query:
                        if re.search(r'\bfemale\b', sentence, re.IGNORECASE):
                            similar_sentences.append(sentence)
                            count += 1
                    else:
                        if not re.search(r'\bfemale\b', sentence, re.IGNORECASE):
                            similar_sentences.append(sentence)
                            count += 1
                    if count == top_k:
                        break

                return similar_sentences, distances[0]

            # Query the database
            query_sentence = f"{Sex} {Age} {Medication} {Disease}"
            # index_path = '/content/drive/My Drive/Programs/BOREALISAI/RAGModel/Data/merged_index_19000.index'
            # metadata_path = '/content/drive/My Drive/Programs/BOREALISAI/RAGModel/Data/merged_metadata_19000.csv'
            similar_sentence, distances = query_database(query_sentence, index_file_path, metadata_file_path, top_k=10)

            # Format the results and store in the dictionary
            formatted_results = "Top 10 most similar sentences with distances:\n"
            formatted_results += "Report ID\tGender\tAge\tSeriousness\tOther Medical Conditions\tHeight\tWeight\tSide Effect Name\tSystem Organ Affected\tMedication\tIndication\tSimilarity Score\n"

            for sentence, distance in zip(similar_sentence, distances):
                formatted_results += f"{sentence}\t{distance}\n"


            # Add to the dictionary under the "Relevant_ADR_reports" key
            results_dict["Relevant_ADR_reports"] = formatted_results

            #CHATGPT QUERY
            openai.api_key = os.getenv("OPENAI_API_KEY")

            # Prepare your question for ChatGPT
            results_summary = (
                f"Clinical Trial Data:\n{results_dict['Clinical_trial_data']}\n\n"
                f"Relevant ADR Reports:\n{results_dict['Relevant_ADR_reports']}\n\n"
                f"ADR Statistics:\n{results_dict['ADR_statistics']}"
            )
            # Format the prompt using an f-string
            prompt = (
                f"A patient wants to know what are the risks of getting an adverse drug reaction to {Medication} taken to treat {Disease}. "
                f"They are a {Age}-year-old {Ethnicity} {Sex}. "
                "To assess the chance that they may have an adverse reaction, please find attached the eligibility criteria for clinical trials for this drug, "
                "the most relevant reported adverse drug reactions, and some statistics about the reports made for this disease. "
                "Based on this information, all trials, and the adverse drug reaction reports, can you tell them what their risk "
                "of having an adverse reaction to this particular drug could be in likelihood based on their race, age, and gender? "
                "When not sure, please say so. When you are confident, please give likelihoods and explanation. "
                "Please always format your response as follows:\n\n"
                
                f"1. Clinical Trial\n"
                "    - Is the patient well represented in the clinical trial for that medication for age, sex and ethnicity, consider all trials?\n\n"
                f"2. Statistics about the recorded side effect\n"
                "    - Look at the ADR Statistic and compare it to the patients characteristics: age and sex\n"
                f"3. Most relevant ADR reports\n"
                "    - look at the most Relevant ADR reports and compare it to the patients characteristics, age and sex\n\n"
                "    - Go into the specifics of what kind of adverse reaction the patients exhibited for that medication\n\n"
                "    - Consider all very relevant reports\n\n"
                "4. Risk Assessment\n"
                "    - Likelihood of ADRs based on the patient's age, ethnicity, and gender\n"
                "    - Any relevant caveats or uncertainties\n\n"
                "5. Recommendation\n"
                "    - Final summary and any suggested actions for the patient\n\n"
            )

            combined_content = f"{prompt}\n\n{results_summary}"

            # Call ChatGPT API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # or "gpt-4" if you're using GPT-4
                messages=[
                    {"role": "system", "content": "You are an assistant."},
                    {"role": "user", "content": combined_content}
                ],
                max_tokens=2000  # Adjust as needed
            )

            # Print the response
            result = response['choices'][0]['message']['content']
            # print("ChatGPT Response: \n\n", answer)

            # Flatten trial_ids and remove any unwanted formatting
            cleaned_trial_ids = []

            for trial_id in trial_ids:
                if isinstance(trial_id, set):  # If the item is a set, convert it to a list and extend
                    cleaned_trial_ids.extend(list(trial_id))
                else:
                    cleaned_trial_ids.append(trial_id)

            # Convert all elements to strings and join without extra characters
            trial_ids_string = ', '.join(cleaned_trial_ids)

            # Print the cleaned result
            part1 = f"\n\nRelevant clinical trials analyzed: {trial_ids_string} are accessible for consultation at https://clinicaltrials.gov/\n"
            part2 = "\nAdverse drug reaction reports have been sourced from the MedEffect Canada database: https://www.canada.ca/en/health-canada/services/drugs-health-products/medeffect-canada.html\n\n"

            result = result + part1 + part2


        elif button_clicked3 == 'risk':
            
            
            #old code starts here
            drug_search = drug_search.upper()
            disease_search = disease_search.upper()

            if disease_search == None or drug_search == None:
                errorFlash = True
                # flash('Please fill in all fields', 'error')
                flash_message_risk = 'Please fill in all fields'
                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html", flash_message_risk=flash_message_risk, user=current_user, text=text, word=word, something=something, meds=meds, errorFlash=errorFlash)    
                return render_template("identify.html", flash_message_risk=flash_message_risk, user=current_user, text=text, word=word, something=something, meds=meds, errorFlash=errorFlash)
                    
            if disease_search == 'SELECT CONDITION':
                errorFlash = True
                # flash('Please select a valid condition', 'error')
                flash_message_risk = 'Please select a valid condition'
                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html",  flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)    
                return render_template("identify.html", flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)

            if drug_search == 'DRUGS IN DATASET' or drug_search == '':
                errorFlash = True
                # flash('Please select a valid condition', 'error')
                flash_message_risk = 'Please select a valid drug'
                user_agent = request.headers.get('User-Agent').lower()
                if 'mobile' in user_agent:
                    return render_template("identify-mobile.html",  flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)    
                return render_template("identify.html", flash_message_risk=flash_message_risk, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, meds=meds)

            prediction_risk = get_model(drug_search, disease_search)
            F = 0
            M = 0

            if prediction_risk is not None:
                R = 0
                M = 100 - prediction_risk
                F = prediction_risk
                

                if current_user.is_authenticated and current_user.sexe is not None:
                    
                    if current_user.sexe.lower() == 'male':

                        if M < F:
                            R = F-M
                            R = str(round(R,2))
                            M = str(round(M,2))
                            result_string = (
                                f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
                                f"Male patients have a <b>{R}%</b> lower risk of developing a reaction compared to female patients. <br>"
                            )
                        else:
                            R = M-F
                            R = str(round(R,2))
                            M = str(round(M,2))
                            result_string = (
                                f"The predicted risk for male patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{M}%</b>. <br><br>"
                                f"Male patients have an additional <b>{R}%</b> risk of developing a reaction compared to female patients.<br> "
                            
                            )
                    elif current_user.sexe.lower() == 'female':
                        if F < M:
                            R = M-F
                            R = str(round(R,2))
                            F = str(round(F,2))
                            result_string = (
                                f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
                                f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients. <br>"
                                
                            )
                        else:
                            R = F-M
                            R = str(round(R,2))
                            F = str(round(F,2))
                            result_string = (
                                f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
                                f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients. <br>"
                            )
                else:
                    if F < M:
                        R = M-F
                        R = str(round(R,2))
                        F = str(round(F,2))
                        result_string = (
                            f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
                            f"Female patients have a <b>{R}%</b> lower risk of developing a reaction compared to male patients.<br>"
                        )
                    else:
                        R = F-M
                        R = str(round(R,2))
                        F = str(round(F,2))
                        result_string = (
                            f"The predicted risk for female patients of developing an adverse drug reaction to {drug_search} given the condition {disease_search} is <b>{F}%</b>. <br><br>"
                            f"Female patients have an additional <b>{R}%</b> risk of developing a reaction compared to male patients.<br>"
                        )
            result_string = result_string

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template("identify-mobile.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, result_string=result_string,meds=meds) 
            return render_template("identify.html", errorFlash=errorFlash, user=current_user, text=text, word=word, something=something, result_string=result_string,meds=meds)
            
        else:
            if button_clicked1 == 'label' and not label_file:
                errorFlash = True
                flash_message_label = 'No file was uploaded'
                # flash('No file was uploaded', 'error')
            if button_clicked2 == 'pill' and not image_file:
                errorFlash = True
                flash_message_pill = 'No file was uploaded'

                # flash('No file was uploaded', 'error')
            if (drug_search == '' or disease_search == '') and button_clicked3 == 'risk':
                # flash('Please fill in all fields', 'error')
                errorFlash = True
                flash_message_risk = 'Please fill in all fields'
                print(drug_search)

        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("identify-mobile.html", flash_message_risk=flash_message_risk, flash_message_pill=flash_message_pill, flash_message_label=flash_message_label, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something,meds=meds)    
        return render_template("identify.html", flash_message_risk=flash_message_risk, flash_message_pill=flash_message_pill, flash_message_label=flash_message_label, errorFlash=errorFlash, user=current_user, text=text, word=word, something=something,meds=meds)

    else:
        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template("identify-mobile.html", errorFlash=errorFlash, user=current_user, word=word, something=something,meds=meds)   
        return render_template("identify.html", errorFlash=errorFlash, user=current_user, word=word, something=something,meds=meds)

@auth.route('/learn')
def learn():
    user_agent = request.headers.get('User-Agent').lower()
    if 'mobile' in user_agent:
        return render_template('learn-mobile.html', user=current_user)
    return render_template('learn.html', user=current_user)
