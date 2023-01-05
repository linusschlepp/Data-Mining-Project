import itertools

import keras

import constants
from main import train_test_split, GaussianNB, accuracy_score, DecisionTreeClassifier, \
    RandomForestClassifier, plt, np, pd, sns, r2_score


def create_outcome_lists(symptoms: list, symptoms_df: pd.DataFrame, neg_value: [bool, str], pos_value: [bool, str]) \
        -> [list, list]:
    """
    Creates two lists, one containing values, where MonkeyPox is True and the other, where MonkeyPox is False

    :param symptoms: List of possible symptoms
    :param symptoms_df: Dataframe where the specific symptom is True and MonkeyPox is Positive
    :param neg_value: Negative value to be searched in dataframe False/ Negative
    :param pos_value: Positive value to be searched in dataframe True/ Positive
    :return: Two lists, one containing the positive values and the other the negative values
    """
    pos_list = [symptoms_df[symptom][neg_value] for symptom in symptoms]
    neg_list = [symptoms_df[symptom][pos_value] for symptom in symptoms]

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
        df[symptom] = symptoms_df.loc[lambda x: x[symptom] == True].groupby(by=constants.MONKEY_POX).sum()[
            symptom].to_dict()

    return df


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data for calculating lift and confidence

    :param data: Complete data, containing symptoms as well as information if patient is positive or negative
    :return: Dataframe, where monkeypox is true, and the corresponding symptoms
    """
    df = data
    df = df.query("{}=='{}'".format(constants.MONKEY_POX, constants.POSITIVE))
    df_concat = pd.concat([pd.DataFrame(columns=list(df[constants.SYSTEMIC_ILLNESS].unique())).fillna(False), df]) \
        .fillna(False)
    for index, col in enumerate(df_concat[constants.SYSTEMIC_ILLNESS]):
        df_concat[col][index] = True
    df_concat = df_concat.drop(columns=[constants.MONKEY_POX, constants.SYSTEMIC_ILLNESS], axis=1)

    return df_concat


def create_feature_accuracy_dict(data_tree: pd.DataFrame, target: pd.DataFrame) -> dict:
    """
    Calculates accuracy for all different feature-combinations and saves them in dictionary

    :param data_tree: Dataframe, on which is operated (All features x)
    :param target: Target data for train_test_split (Outcome y)
    :return: A dictionary, containing different combination of symptoms as key and the corresponding accuracies as value
    """
    sol_dict = {}
    combinations_col = create_combination_list(len(data_tree.columns))
    models = [GaussianNB(), DecisionTreeClassifier(criterion='entropy', splitter='best',
                                                   min_samples_split=5), RandomForestClassifier(n_estimators=100)]

    for combination in combinations_col:
        temp_data = data_tree.copy()
        # Drop all columns, which are not contained within combination
        temp_data.drop(columns=temp_data.columns.difference(list(combination)), inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(temp_data, target)
        for model in models:
            model.fit(X_train, y_train)
            y_prediction = model.predict(X_test)
            accuracy = (100 * accuracy_score(y_test, y_prediction))
            r2_test = model.score(X_test, y_test)
            r2_train = model.score(X_train, y_train)
            sol_dict[(str(model), str(temp_data.columns.values.tolist()))] = (accuracy, r2_test, r2_train)

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


def create_combination_list(am_columns: int) -> list:
    """
    Creates a list, containing all possible feature-combinations, by concatenating lists with different sizes

    :param am_columns: Amount of columns in dataframe
    :return: List, containing all possible feature-combinations
    """
    # TODO: Try to use list comprehension
    ret_list = []
    for x in range(2, am_columns):
        ret_list += list(itertools.combinations(am_columns, x))

    return ret_list


def check_over_fitting(data: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    Checks data for overfitting, by creating a plot. If train-plot is higher than test-plot, risk of overfitting
    could exist within data.

    :param data: Data to be checked (All features x)
    :param target: Target-data to be checked (Outcome y)
    """
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.5)
    values = [i for i in range(2, 21)]
    train_scores = [fetch_score(x_train, y_train, value) for value in values]
    test_scores = [fetch_score(x_test, y_test, value) for value in values]
    plt.plot(values, train_scores, '-o', label='Train', color=constants.FIRST_COLOR)
    plt.plot(values, test_scores, '-o', label='Test', color=constants.SECOND_COLOR)
    plt.xlabel('Depth of tree')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_data(symptoms: list, pos_lst: list, neg_lst: list, x_text: str, y_text: str, legend: str,
              text_legend_1: [bool, str], text_legend_2: [bool, str]) -> None:
    """
    Plots positive and negative features as barplot.

    :param symptoms: List of symptoms as str
    :param pos_lst: List of positive symptoms
    :param neg_lst: List of negative symptoms
    :param x_text: Text of the x-axis
    :param y_text: Text of the y-axis
    :param text_legend_1: Text of legend value 1
    :param text_legend_2: Text of legend value 2
    :param legend: Caption of the legend of the plot
    """
    df1 = pd.DataFrame({x_text: symptoms, y_text: pos_lst})
    df2 = pd.DataFrame({x_text: symptoms, y_text: neg_lst})
    df1[legend] = text_legend_1
    df2[legend] = text_legend_2
    res = pd.concat([df1, df2])
    sns.barplot(x=x_text, y=y_text, data=res, hue=legend, palette=constants.COLOR_PALETTE)
    plt.show()


def fetch_score(x: pd.DataFrame, y: pd.DataFrame, value: int) -> float:
    """
    Used for checking for overfitting.  Calculates accuracy

    :param x: Data to be checked (All features x)
    :param y: Target data to be checked (Outcome y)
    :param value: Value in the range of 1 to 21
    :return: Calculating accuracy corresponding to model DecisionTreeClassifier and the max_depth of value
    """
    model = RandomForestClassifier(max_depth=value)
    model.fit(x, y)
    yhat = model.predict(x)
    acc = accuracy_score(y, yhat)

    return acc


def prepare_for_tensor(data_x: pd.DataFrame, target: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """
    Prepares feature and target dataset for tensorflow. All values within those dataframes are changed to the
    corresponding float values.

    :param data_x: Data to be checked (All features x)
    :param target: Target data to be checked (Outcome y)
    :return: Feature and target dataset, where values are changed to float
    """
    data_x = np.asarray(data_x).astype('float32')
    target = target.replace([constants.POSITIVE, constants.NEGATIVE], [True, False])
    target = np.asarray(target).astype('float32')

    return data_x, target


def eval_data(X: pd.DataFrame, y: pd.DataFrame, model: keras.models.Sequential, calculate_r2: bool = False) -> None:
    """
    Evaluates train- and test, corresponding to given model

    :param X:  to be checked (All features x)
    :param y: Target data to be checked (Outcome y)
    :param model: Neuronal Network Model, on which is to be operated
    :param calculate_r2: bool-flag if r² has to be calculated for the given data, is per default False
    """
    solution = model.evaluate(X, y, verbose=0)
    y_prediction = model.predict(X)
    print('{}: {:.2f}\n{}: {:.2f}'.format(model.metrics_names[0], solution[0],
                                          model.metrics_names[1], solution[1]))
    if calculate_r2:
        print('R²: {:.2f}'.format(r2_score(y, y_prediction)))
