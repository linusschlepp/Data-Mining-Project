import pandas as pd
from main import train_test_split, GaussianNB, accuracy_score
import yaml


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


def create_feature_accuracy_dict(data_tree, target, list_symptoms):
    """
    Calculates accuracy for different feature combination and saves them in dictionary

    :param data_tree:
    :param target: target data for train_test_split
    :param list_symptoms: List of symptoms/features
    :return: A dictionary, containing different combination of symptoms as key and the corresponding accuracy as value
    """
    sol_dict = {}
    ran_stream = 23
    for symptom in list_symptoms:
        temp_df = data_tree.drop([symptom], axis= 1)
        x_train, x_test, y_train, y_test = train_test_split(temp_df, target, random_state=ran_stream)
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_predicion = model.predict(x_test)
        accuracy = (100 * accuracy_score(y_test, y_predicion))
        sol_dict[str(temp_df.columns.values.tolist())] = accuracy

    return sol_dict


