# File contains all utils functions, used in this project

import itertools
import keras
import constants
from main import train_test_split, GaussianNB, accuracy_score, DecisionTreeClassifier, \
    RandomForestClassifier, plt, np, pd, sns


def create_outcome_lists(symptoms: list, symptoms_df: pd.DataFrame, neg_value: [bool, str], pos_value: [bool, str]) \
        -> [list, list]:
    """
    Creates two lists, one containing all positive values of feature/ symptom  and the other containing all negative
    values of the feature/ symptom

    :param symptoms: List of symptoms/ features
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
    Fetches True-values from given dataframe

    :param symptoms: List of symptoms/ features
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
    Cleans data for the purpose of calculating lift and confidence

    :param data: Complete data, containing all features and solution column
    :return: Dataframe, where monkeypox is True, and the corresponding symptoms
    """
    df = data
    # Only get rows where MonkeyPox is positive
    df = df.query("{}=='{}'".format(constants.MONKEY_POX, constants.POSITIVE))
    # Concat list of Systemic illnesses with data substitute nan values with False
    df_concat = pd.concat([pd.DataFrame(columns=list(df[constants.SYSTEMIC_ILLNESS].unique())).fillna(False), df]) \
        .fillna(False)
    # Iterate through Systemic Illness column if illness occurs change row in corresponding column to True
    for index, col in enumerate(df_concat[constants.SYSTEMIC_ILLNESS]):
        df_concat[col][index] = True

    # Drop MonkeyPox and Systemic Illness column, they are not relevant for calculating lift and confidence
    df_concat = df_concat.drop(columns=[constants.MONKEY_POX, constants.SYSTEMIC_ILLNESS], axis=1)

    return df_concat


def create_feature_accuracy_dict(X_data: pd.DataFrame, target: pd.DataFrame) -> dict:
    """
    Calculates accuracy for all different feature-combinations and models, and stores them into dictionary

    :param X_data: Dataframe, on which is operated (All features x)
    :param target: Target data for train_test_split (Outcome y)
    :return: A dictionary, containing different combination of symptoms as key and the corresponding accuracies as value
    """
    sol_dict = {}
    # List, containing all possible feature-combinations
    combinations = create_combination_list(X_data.columns)
    # List, containing all models, used in observation
    models = [GaussianNB(), DecisionTreeClassifier(criterion='entropy', splitter='best',
                                                   min_samples_split=5), RandomForestClassifier(n_estimators=100)]

    for combination in combinations:
        # Copy data to temporary dataframe
        temp_data = X_data.copy()
        # Drop all columns, which are not contained within combination
        temp_data.drop(columns=temp_data.columns.difference(list(combination)), inplace=True)
        # Get train and test data
        X_train, X_test, y_train, y_test = train_test_split(temp_data, target)
        # Iterate through models and use combination from outer-for-loop for calculating accuracy
        for model in models:
            model.fit(X_train, y_train)
            y_prediction = model.predict(X_test)
            accuracy = (100 * accuracy_score(y_test, y_prediction))
            # Save feature-combination and used model as key in dict and accuracy as value
            sol_dict[(str(model), str(temp_data.columns.values.tolist()))] = accuracy

    return sol_dict


def create_combination_list(columns: pd.core) -> list:
    """
    Creates a list, containing all possible feature-combinations

    :param columns: Columns of dataframe
    :return: List, containing all possible feature-combinations
    """
    ret_list = []
    # Concatenate combination-list, starting by feature-combination length 2
    for x in range(2, len(columns)):
        ret_list += list(itertools.combinations(columns, x))

    return ret_list


def check_over_fitting(data: pd.DataFrame, target: pd.DataFrame) -> None:
    """
    Checks data for overfitting by creating a plot. If train-plot is higher than test-plot, risk of overfitting
    could exist within data.

    :param data: Data to be checked (All features x)
    :param target: Target-data to be checked (Outcome y)
    """
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.5)
    tree_depths = [i for i in range(2, 21)]
    train_scores = [fetch_score(x_train, y_train, tree_depth) for tree_depth in tree_depths]
    test_scores = [fetch_score(x_test, y_test, tree_depth) for tree_depth in tree_depths]
    plt.plot(tree_depths, train_scores, '-o', label='Train', color=constants.FIRST_COLOR)
    plt.plot(tree_depths, test_scores, '-o', label='Test', color=constants.SECOND_COLOR)
    plt.xlabel('Depth of tree')
    plt.ylabel('Accuracy')
    plt.title('Accuracy history of test and training data by using RandomForestClassifier')
    plt.legend()
    plt.show()


def plot_data(symptoms: list, pos_lst: list, neg_lst: list, x_text: str, y_text: str, legend: str,
              text_legend_1: [bool, str], text_legend_2: [bool, str], title: str = '') -> None:
    """
    Plots positive and negative features as bar plot

    :param symptoms: List of symptoms as str
    :param pos_lst: List of True symptoms
    :param neg_lst: List of False symptoms
    :param x_text: Text of the x-axis
    :param y_text: Text of the y-axis
    :param legend: Caption of the legend of the plot
    :param text_legend_1: Text of legend value 1
    :param text_legend_2: Text of legend value 2
    :param title: Title of plot
    """
    df1 = pd.DataFrame({x_text: symptoms, y_text: pos_lst})
    df2 = pd.DataFrame({x_text: symptoms, y_text: neg_lst})
    df1[legend] = text_legend_1
    df2[legend] = text_legend_2
    res = pd.concat([df1, df2])
    sns.barplot(x=x_text, y=y_text, data=res, hue=legend, palette=constants.COLOR_PALETTE)
    plt.title(title)
    plt.show()


def fetch_score(x: pd.DataFrame, y: pd.DataFrame, tree_depth: int) -> float:
    """
    Used for checking overfitting. Calculates accuracy for given tree depth

    :param x: Data to be checked (All features x)
    :param y: Target-data to be checked (Outcome y)
    :param tree_depth: Value in the range of 1 to 21 (tree depth)
    :return: Calculated accuracy corresponding to model RandomForestClassifier and the max_depth of value
    """
    model = RandomForestClassifier(max_depth=tree_depth)
    model.fit(x, y)
    yhat = model.predict(x)
    # Calculate accuracy
    acc = accuracy_score(y, yhat)

    return acc


def prepare_for_tensor(data_x: pd.DataFrame, target: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """
    Prepares feature and target dataset for tensorflow. All values within those dataframes are changed to their
    corresponding float values.

    :param data_x: Data to be checked (All features x)
    :param target: Target-data to be checked (Outcome y)
    :return: Feature and target dataset, where values are changed to float
    """
    data_x = np.asarray(data_x).astype('float32')
    target = target.replace([constants.POSITIVE, constants.NEGATIVE], [True, False])
    target = np.asarray(target).astype('float32')

    return data_x, target


def eval_data(X: pd.DataFrame, y: pd.DataFrame, model: keras.models.Sequential) -> None:
    """
    Evaluates train- and test, corresponding to given model

    :param X: Data  to be checked (All features x)
    :param y: Target-data to be checked (Outcome y)
    :param model: Neuronal Network Model, on which is to be operated
    """
    # Evaluate model
    solution = model.evaluate(X, y, verbose=0)
    # Print solution
    print('{}: {:.2f}\n{}: {:.2f}'.format(model.metrics_names[0], solution[0],
                                          model.metrics_names[1], solution[1]))

