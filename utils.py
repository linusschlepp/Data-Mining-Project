import pandas as pd


def create_outcome_list(symptoms, df):
    pos_list = [df[symptom]['Positive'] for symptom in symptoms]
    neg_list = [df[symptom]['Negative'] for symptom in symptoms]

    return pos_list, neg_list


def make_df_true(symptoms, symptoms_df):
    df = pd.DataFrame({})
    for symptom in symptoms:
        df[symptom] = symptoms_df.loc[lambda x: x[symptom] == True].groupby(by='MonkeyPox').sum()[symptom].to_dict()

    return df
