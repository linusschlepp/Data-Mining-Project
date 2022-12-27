import itertools


import constants
from main import train_test_split, GaussianNB, accuracy_score, combinations, DecisionTreeClassifier, \
    RandomForestClassifier, pyplot, np,pd
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
    df_concat = pd.concat([pd.DataFrame(columns=list(df[constants.SYSTEMIC_ILLNESS].unique())).fillna(False), df]) \
        .fillna(False)
    for index, col in enumerate(df_concat[constants.SYSTEMIC_ILLNESS]):
        df_concat[col][index] = True
    df_concat = df_concat.drop(columns=[constants.MONKEY_POX, constants.SYSTEMIC_ILLNESS], axis=1)

    return df_concat


def create_feature_accuracy_dict(data_tree: pd.DataFrame, target: pd.DataFrame) -> dict:
    """
    Calculates accuracy for all different feature-combinations and saves them in dictionary

    :param data_tree: Dataframe, on which is operated
    :param target: target data for train_test_split
    :return: A dictionary, containing different combination of symptoms as key and the corresponding accuracy as value
    """
    sol_dict = {}
    ran_stream = 23
    combinations_col = create_combination_list(data_tree.columns)  # list(itertools.combinations(data_tree.columns, 3))
    models = [GaussianNB(), DecisionTreeClassifier(criterion='entropy', splitter='best',
                                                   min_samples_split=5), RandomForestClassifier(n_estimators=100)]

    for combination in combinations_col:
        temp_data = data_tree.copy()
        temp_data.drop(columns=temp_data.columns.difference(list(combination)), inplace=True)
        x_train, x_test, y_train, y_test = train_test_split(temp_data, target, random_state=ran_stream)
        for model in models:
            model.fit(x_train, y_train)
            y_prediction = model.predict(x_test)
            accuracy = (100 * accuracy_score(y_test, y_prediction))
            sol_dict[(str(model), str(temp_data.columns.values.tolist()))] = accuracy

    # TODO: Try to solve with dict comprehensiont
    # sol_dict = {
    #     outer_k : {
    #         inner_k: prepare_train_test_data(data_tree, co)
    #         for inner_k, inner_v in outer_v.items()
    #     }
    #     for outer_k, outer_v in outer_dict.items()
    # }
    #
    return sol_dict


def prepare_train_test_data(data_tree, combination, target):
    ran_stream = 23
    temp_data = data_tree.copy()
    temp_data.drop(columns=temp_data.columns.difference(list(combination)), inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(temp_data, target, random_state=ran_stream)

    return x_train, x_test, y_train, y_test


def create_combination_list(data_columns) -> list:
    """
    Creates a list, containing all possible feature-combinations, by concatenating lists with different sizes

    :param data_columns: Columns within feature data
    :return: List, containing all possible feature-combinations
    """
    #TODO: Try to use list comprehension
    ret_list = []
    for x in range(2, len(data_columns)):
        temp_lst = list(itertools.combinations(data_columns, x))
        ret_list += temp_lst

    return ret_list


def convert_dict_to_json(dict_to_convert: dict):
    """
    Converts given dictionary to json-object.

    :param dict_to_convert: Dictionary, to be converted
    :return: Dictionary as json-object
    """
    return json.dumps(dict_to_convert, sort_keys=True, indent=4)


def check_over_fitting(data: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    Checks data for overfitting, by creating a plot, if test-plot is higher than train-plot. Risk of overfitting is high

    :param data: Data to be checked (All features x)
    :param target: Target-data to be checked (Outcome y)
    """
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3)
    values = [i for i in range(2, 21)]
    print(data.shape, target.shape)
    train_scores = [fetch_score(x_train, y_train, value) for value in values]
    test_scores = [fetch_score(x_test, y_test, value) for value in values]
    pyplot.plot(values, train_scores, '-o', label='Train')
    pyplot.plot(values, test_scores, '-o', label='Test')
    pyplot.xlabel('Depth of tree')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.show()


def fetch_score(x: pd.DataFrame, y: pd.DataFrame, value: int) -> float:
    """
    Used for checking for overfitting.  Calculates accuracy

    :param x: All features
    :param y: Target data
    :param value: Value in the range of 1 to 21
    :return: Calculating accuracy corresponding to model DecisionTreeClassifier and the max_depth of value
    """
    model = DecisionTreeClassifier(max_depth=value)
    model.fit(x, y)
    yhat = model.predict(x)
    acc = accuracy_score(y, yhat)

    return acc


def prepare_for_tensor(data_x: pd.DataFrame, target: pd.DataFrame):
    """
    Prepares feature and target dataset for tensorflow. Meaning all values within those dataframes are changed to float

    :param data_x: All features
    :param target: Target data
    :return: Feature and target dataset, where values are changed to float
    """
    data_x = np.asarray(data_x).astype('float32')
    target = target.replace(['Positive', 'Negative'], [True, False])
    target = np.asarray(target).astype('float32')

    return data_x, target
