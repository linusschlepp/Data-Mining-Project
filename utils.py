import pandas as pd


def create_outcome_lists(symptoms, symptoms_df):
    """
    Creates two lists, one containing values, where MonkeyPox is True and the other, where MonkeyPox is False

    :param symptoms: List of possible symptoms
    :param symptoms_df: Dataframe where the specific symptom is True and MonkeyPox is Positive
    :return: Two lists, one list contains the positive values and the other the negative ones
    """
    pos_list = [symptoms_df[symptom]['Positive'] for symptom in symptoms]
    neg_list = [symptoms_df[symptom]['Negative'] for symptom in symptoms]

    return pos_list, neg_list


def fetch_true_symptoms(symptoms, symptoms_df):
    """
    Fetches True-values from given dataframe: symptoms_df

    :param symptoms: List of possible symptoms
    :param symptoms_df: Dataframe, storing all values of symptoms
    :return: Dataframe, containing all symptoms, which are true
    """
    df = pd.DataFrame({})
    for symptom in symptoms:
        df[symptom] = symptoms_df.loc[lambda x: x[symptom] == True].groupby(by='MonkeyPox').sum()[symptom].to_dict()

    return df
