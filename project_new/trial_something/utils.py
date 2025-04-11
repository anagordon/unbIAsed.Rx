# filepath: c:\Backup\OneDrive\Documents\GitHub\unbIAsed.Rx\project_new\trial_something\utils.py

import os
import pandas as pd
import pickle
import numpy as np
from .views import get_study_ids, get_study, sort  # Import other helper functions if needed

def get_model(drug, disease):
    model_file_path = os.path.join(os.path.dirname(__file__), 'regression_model.pkl')
    preprocessor_file_path = os.path.join(os.path.dirname(__file__), 'preprocessor.pkl')

    BASE_DIR1 = os.path.dirname(os.path.abspath(__file__))
    csv_file1 = os.path.join(BASE_DIR1, 'ctg-studies.csv')
    df1 = pd.read_csv(csv_file1)

    df1["Conditions"] = df1["Conditions"].str.upper()
    df1["Interventions"] = df1["Interventions"].str.upper()
    df1["Conditions"] = df1["Conditions"].fillna('').astype(str)
    df1["Interventions"] = df1["Interventions"].fillna('').astype(str)

    with open(preprocessor_file_path, 'rb') as f:
        preprocessor = pickle.load(f)

    with open(model_file_path, 'rb') as f:
        model = pickle.load(f)

    nct_list = get_study_ids(drug, disease, df1)
    num_studies, female_proportion, male_proportion, tot_num_females, tot_num_males = get_study(nct_list)

    (top_3, total_participants, sum_females, sum_males, female_proportion1, male_proportion1) = sort(nct_list)

    sample_data = pd.DataFrame({
        'Indication': [disease],
        'Num Studies': [num_studies],
        'Total females in studies': [tot_num_females],
        'Total males in studies': [tot_num_males],
        'Male proportion in studies': [male_proportion],
        'Female proportion in studies': [female_proportion],
        'Number of participants in most relevant studies': [total_participants],
        'Number of female participants in most relevant studies': [sum_females],
        'Number of male participants in most relevant studies': [sum_males],
        'Proportion of females in most relevant studies': [female_proportion1],
        'Proportion of males in most relevant studies': [male_proportion1]
    })

    transformed_data = preprocessor.transform(sample_data)

    transformed_columns = (
        preprocessor.transformers_[0][1].get_feature_names_out(['Indication']).tolist() +
        [
            'Total females in studies',
            'Total males in studies',
            'Female proportion in studies',
            'Male proportion in studies',
            'Proportion of females in most relevant studies',
            'Proportion of males in most relevant studies',
            'Num Studies',
            'Number of participants in most relevant studies',
            'Number of female participants in most relevant studies',
            'Number of male participants in most relevant studies'
        ]
    )

    transformed_df = pd.DataFrame(transformed_data, columns=transformed_columns)
    transformed_df.drop(columns=['Female proportion in studies', 'Male proportion in studies', 'Proportion of females in most relevant studies', 'Proportion of males in most relevant studies'], inplace=True)
    prediction = model.predict(transformed_df) * 100
    return np.around(prediction[0], 2)
