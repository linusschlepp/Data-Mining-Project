def create_outcome_list(symptoms, df):
    pos_list = [df[symptom]['Positive'] for symptom in symptoms ]
    neg_list = [df[symptom]['Negative'] for symptom in symptoms ]

    return pos_list, neg_list
