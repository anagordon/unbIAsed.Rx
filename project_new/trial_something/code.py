from flask import Blueprint, render_template, request
import openai
import json
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AutoTokenizer, AutoModel
import torch
import pandas as pd
import os
import numpy as np
import seaborn as sns
import tensorflow_hub as hub
from flask_login import current_user
import faiss
# import torch.nn as nn
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time

# Set environment variables to resolve OpenMP runtime conflict and disable oneDNN custom operations
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OPENAI_API_KEY"] = "" #make sure to insert your own API key here
os.environ["FLASK_APP"] = "main.py"

code = Blueprint('code', __name__)

# Function to create a requests session with retries
def requests_retry_session(
    retries=5,
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


@code.route('/identify/new-model', methods=['GET', 'POST'])
def new_model():
    #Write Code Here
    try:
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
            # response = requests.get(url, params=payload)
            try:
                response = requests.get(url, params=payload, timeout=10)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                return "The request to the external API timed out. Please try again later.", 504
            except requests.exceptions.RequestException as e:
                print(f"API Error: {e}")
                return "An error occurred while fetching data from the external API.", 500


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
                print("Request was successful.")
                try:
                    data = response.json()
                    print("Data received from API:", data)
                    studies = data.get('studies', [])
                    print("Number of studies found:", len(studies))


                    # Collect clinical trial data as a string
                    clinical_trial_data = []
                    for idx, study_data in enumerate(studies):
                        print(f"Processing study {idx + 1}...")
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
                            print(f"Error processing study {idx + 1}: Missing expected attributes.")
                            continue
                    results_dict["Clinical_trial_data"] = ''.join(clinical_trial_data)  # Save clinical trial data as string
                    print("Clinical trial data collected successfully.")
                except requests.exceptions.JSONDecodeError:
                    print("Failed to decode JSON. Here is the raw response:")
                    # print(response.text)
                    return "An error occurred while processing the JSON response.", 500
            else:
                print(f"Request failed with status code: {response.status_code}")
                # print(response.text)

            #GET THE ADR REPORT STATISTICS FOR SPECIFIC MEDICATION
            # GET THE ADR REPORT STATISTICS FOR SPECIFIC MEDICATION
            try:
                # Define the path to the ADRdata.csv file
                csv_file_path = os.path.join(os.path.dirname(__file__), 'ADRdata.csv')
                print("CSV file path: ", csv_file_path)

                # Check if the file exists
                if not os.path.exists(csv_file_path):
                    print(f"CSV file not found: {csv_file_path}")
                    return "Required ADR data file is missing.", 500

                # Load the CSV file
                df = pd.read_csv(csv_file_path, delimiter='\t')
                print("ADR data loaded successfully. DataFrame shape:", df.shape)

                # Check for required columns
                required_columns = ['DRUGNAME', 'AGE_Y', 'GENDER_ENG', 'SERIOUSNESS_ENG']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"Missing required columns in ADR data: {missing_columns}")
                    return f"Missing required columns in ADR data: {', '.join(missing_columns)}", 500

                # Filter rows where the DRUGNAME column contains the specific string
                specific_string = Medication.upper()
                filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]
                print(f"Filtered DataFrame shape: {filtered_df.shape}")

                if filtered_df.empty:
                    print("No ADR data found for the specified medication.")
                    return "No ADR data found for the specified medication.", 404

            except Exception as e:
                print(f"Error loading ADR data: {e}")
                return "An error occurred while loading ADR data.", 500
          
            print("CSV file path: ", csv_file_path)
            print("Medication: ", Medication)

            df = pd.read_csv(csv_file_path, delimiter='\t')
            print("CSV file loaded successfully.")
            print("DataFrame shape: ", df.shape)

            # Define the specific string you're looking for
            specific_string = Medication.upper()

            # Filter rows where the DRUGNAME column contains the specific string
            filtered_df = df[df['DRUGNAME'].str.contains(specific_string, case=False, na=False)]
            print(f"Filtered DataFrame shape: {filtered_df.shape}")
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
            print("Results dictionary updated with ADR statistics: ", results_dict["ADR_statistics"])
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
                print("Sentence embeddings generated successfully.")

                    
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
                print("Query executed successfully.")
                print("Similar sentences:", similar_sentences)
                print("Distances:", distances[0])

            # Query the database
            query_sentence = f"{Sex} {Age} {Medication} {Disease}"
            print("Querying database with sentence:", query_sentence)
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
            print("Relevant ADR reports collected successfully.")

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
            print("ChatGPT Response: \n\n", result)
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
            print("Final Result: \n\n", result)

            user_agent = request.headers.get('User-Agent').lower()
            if 'mobile' in user_agent:
                return render_template('new_model-mobile.html', user=current_user, result=result)
            return render_template('new_model.html', user=current_user, result=result)
    
        user_agent = request.headers.get('User-Agent').lower()
        if 'mobile' in user_agent:
            return render_template('new_model-mobile.html', user=current_user)
        return render_template('new_model.html', user=current_user)

    except Exception as e:
        # Log the error and return a user-friendly message
        print(f"Error: {e}")
        return "An error occurred while processing your request.", 500