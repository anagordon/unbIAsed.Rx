from flask import Blueprint, render_template, request, jsonify, flash, redirect, session, url_for
from flask_login import login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, ValidationError
from wtforms.validators import DataRequired, Email
from sqlalchemy import or_
from fuzzywuzzy import process
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer, AutoModel
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from albumentations import (
    Compose, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    Rotate, ShiftScaleRotate, Transpose
)
from albumentations.pytorch import ToTensorV2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .models import User, Drugs
from . import db
import openai
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import faiss
import seaborn as sns
import tensorflow_hub as hub
import pytesseract
import cv2
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os
import requests
import json
import difflib
from transformers.utils import logging

# Download NLTK stopwords
nltk.download('stopwords')


# Set environment variables to resolve OpenMP runtime conflict and disable oneDNN custom operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENAI_API_KEY"] = "" #make sure to insert your own API key here

code = Blueprint('code', __name__)

# Function to create a requests session with retries
def requests_retry_session(
    retries=10,
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


@code.route('/new-model', methods=['GET', 'POST'])
def new_model():
    #Write Code Here
    result = ""

    if request.method == 'POST':
        # Variables
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
        
        chunk_size = 10000  # Number of rows per chunk
        filtered_chunks = []
        specific_string = Medication.upper()


        print("Loading CSV file in chunks...")
        for chunk in pd.read_csv(csv_file_path, delimiter='\t', chunksize=chunk_size):
            # Filter rows in the current chunk
            filtered_chunk = chunk[chunk['DRUGNAME'].str.contains(specific_string, case=False, na=False)]
            filtered_chunks.append(filtered_chunk)

        # Combine all filtered chunks into a single DataFrame
        filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Filtered DataFrame shape: {filtered_df.shape}")

        # df = pd.read_csv(csv_file_path, delimiter='\t')

        # Define the specific string you're looking for
        # specific_string = Medication.upper()

        # Filter rows where the DRUGNAME column contains the specific string
        # filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]

        # ---- GET AGE INFORMATION -----
        # Define age bins and labels for grouping
        age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float('inf')]
        age_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100+']

        # Categorize ages into bins
        filtered_df['AGE_GROUP'] = pd.cut(filtered_df['AGE_Y'], bins=age_bins, labels=age_labels, right=False)

        # Count occurrences for each age group and calculate percentages
        age_group_counts = filtered_df['AGE_GROUP'].value_counts().sort_index()
        age_group_percentages = (age_group_counts / age_group_counts.sum()) * 100

        # Select the top 3 age groups by count
        top_3_age_groups = age_group_percentages.nlargest(3).index.astype(str).tolist()
        age_group_sentence = f"The patients with the highest rate of reported drug reaction for this medicine are \"{top_3_age_groups[0]}\" then \"{top_3_age_groups[1]}\" then \"{top_3_age_groups[2]}\"."

        # ----- GET SEX INFORMATION ------
        gender_counts = filtered_df['GENDER_ENG'].value_counts()
        gender_percentages = (gender_counts / gender_counts.sum()) * 100

        female_percentage = gender_percentages.get('Female', 0)
        male_percentage = gender_percentages.get('Male', 0)
        gender_sentence = f"Out of all the reports, {female_percentage:.2f}% were women's and {male_percentage:.2f}% were men's."

        # ----- GET SERIOUSNESS INFORMATION -----
        seriousness_counts = filtered_df['SERIOUSNESS_ENG'].value_counts()
        seriousness_percentages = (seriousness_counts / seriousness_counts.sum()) * 100

        serious_percentage = seriousness_percentages.get('Serious', 0)
        not_serious_percentage = seriousness_percentages.get('Not Serious', 0)
        seriousness_sentence = f"Out of all the reported ADRs, {serious_percentage:.2f}% were serious and {not_serious_percentage:.2f}% were not serious."

        # Combine all sentences into one string and save to ADR_statistics key
        results_dict["ADR_statistics"] = f"{age_group_sentence}\n{gender_sentence}\n{seriousness_sentence}"
        print(results_dict["ADR_statistics"])
        #GET MOST RELEVANT ADR REPORTS
        #Load BERT model and tokenizer
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', output_hidden_states=True)
        logging.set_verbosity_info()  # Set logging level to INFO
        model_str = "NeuML/pubmedbert-base-embeddings"
        tokenizer = AutoTokenizer.from_pretrained(model_str)
        model = AutoModel.from_pretrained(model_str)
        # print("Model and tokenizer loaded successfully.")
        # try:
        #     print("Loading tokenizer...")
        #     tokenizer = AutoTokenizer.from_pretrained(model_str)
        #     print("Loading model...")
        #     model = AutoModel.from_pretrained(model_str)
        #     print("Model and tokenizer loaded successfully.")
        # except Exception as e:
        #     print(f"Error loading model or tokenizer: {e}")
        #     return "An error occurred while loading the model or tokenizer.", 500

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
        print("FAISS index loaded successfully.")

        metadata_file_path = os.path.join(os.path.dirname(__file__), 'merged_metadata_19000.csv')
        metadata_df = pd.read_csv(metadata_file_path)
        print("Metadata loaded successfully.")
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
        print("Querying the database...")
        print(f"Query sentence: {query_sentence}")
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
        print(result)
        return render_template('new_model.html', result=result, user =current_user)
    
    return render_template('new_model.html', user =current_user)

