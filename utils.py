import pandas as pd

import constants
from main import train_test_split, GaussianNB, accuracy_score
import json


def create_outcome_lists(symptoms: list, symptoms_df: pd.DataFrame):
    """
    Creates two lists, one containing values, where MonkeyPox is True and the other, where MonkeyPox is False

    :param symptoms: List of possible symptoms
    :param symptoms_df: Dataframe where the specific symptom is True and MonkeyPox is Positive
    :return: Two lists, one list contains the positive values and the other the negative ones
    """
    pos_list = [symptoms_df[symptom]['Positive'] for symptom in symptoms]
    neg_list = [symptoms_df[symptom]['Negative'] for symptom in symptoms]

    return pos_list, neg_list


def fetch_true_symptoms(symptoms: list, symptoms_df: pd.DataFrame) -> pd.DataFrame():
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


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data for calculating lift and confidence

    :param data: Complete data, containing symptoms as well as information if patient is positive or negative
    :return: Dataframe, where monkeypox is true, and the corresponding symptoms
    """
    df = data
    df = df.query("{}=='Positive'".format(constants.MONKEY_POX))
    # Change all systemic illnesses to true
    df = df.replace(list(df[constants.SYSTEMIC_ILLNESS].unique()), True)
    df = df.drop(columns=[constants.MONKEY_POX], axis=1)

    return df


def create_feature_accuracy_dict(data_tree: pd.DataFrame, target: pd.DataFrame, list_symptoms: list) -> dict:
    """
    Calculates accuracy for different feature combination and saves them in dictionary

    :param data_tree: Dataframe, on which is operated
    :param target: target data for train_test_split
    :param list_symptoms: List of symptoms/features
    :return: A dictionary, containing different combination of symptoms as key and the corresponding accuracy as value
    """
    sol_dict = {}
    ran_stream = 23
    for symptom in list_symptoms:
        temp_df = data_tree.drop([symptom], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(temp_df, target, random_state=ran_stream)
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_predicion = model.predict(x_test)
        accuracy = (100 * accuracy_score(y_test, y_predicion))
        sol_dict[str(temp_df.columns.values.tolist())] = accuracy

    return sol_dict


def convert_dict_to_json(dict_to_convert: dict):
    """
    Converts given dictionary to json-object.

    :param dict_to_convert: Dictionary, to be converted
    :return: Dictionary as json-object
    """
    return json.dumps(dict_to_convert, sort_keys=True, indent=4)
