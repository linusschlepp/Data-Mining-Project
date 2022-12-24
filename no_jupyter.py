import operator

import utils
from main import *
import constants
import pprint

pp = pprint.PrettyPrinter(indent=4)

# test_dict = dict.fromkeys(data[constants.SYSTEMIC_ILLNESS].unique())
# #sns.pairplot(data.head(100), hue='MonkeyPox')
# for index in data.index:
#     if data['MonkeyPox'][index] == 'Positive':
#         test_dict[data[constants.SYSTEMIC_ILLNESS][index]] = int(
#             test_dict[data[constants.SYSTEMIC_ILLNESS][index]] or 0) + 1
#
# print(test_dict)
# df = pd.DataFrame(test_dict.items())
#
# df.rename(columns={0: 'illness', 1: 'quantity'}, inplace=True)
# plt.bar(df['illness'], df['quantity'])
# plt.title('Cases of monkeypox seperated by systemic illness')
# plt.xlabel('Illness')
# plt.ylabel('Occurrence')
# plt.show()
# simple_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
# data[constants.SYSTEMIC_ILLNESS] = [simple_dict[item] for item in data[constants.SYSTEMIC_ILLNESS]]
# data_tree = data.iloc[:, :-1]
# target = data.iloc[:, -1:]
# ran_stream = 23
# x_train, x_test, y_train, y_test = train_test_split(data_tree, target, random_state=ran_stream)
#
# model = GaussianNB()
# # x_train.loc[x_train['MonkeyPox'] == 'Negative'] = False
# # x_train.loc[x_train['MonkeyPox'] == 'Positive'] = True
# # x_train['MonkeyPox'] = x_train['MonkeyPox'].astype(int)
# # model.fit(y_train, x_train)
# model.fit(x_train, y_train)
# y_prediction = model.predict(x_test)
# print(100 * accuracy_score(y_test, y_prediction))
# # x_train, x_test, y_train, y_test = train_test_split(data_tree, target, random_state=ran_stream, test_size=0.01)
# # model_tree = DecisionTreeClassifier(criterion='entropy', splitter='best')
# # model_tree.fit(x_train, y_train)
# # text_representation = export_text(model_tree, feature_names=data_tree.columns.values.tolist())
# # figure = plt.figure(figsize=(10,8))
# # plot_tree(model_tree, feature_names=data_tree.columns, filled=True, rounded=True)
# # plt.show()
#
# series = data['MonkeyPox'].value_counts()
# plt.bar(series.index, series.values)
# plt.show()
#
#
# symptoms = data.iloc[:, 1:-1].columns.values.tolist()
# symptoms_df = data.iloc[:, 1:]
# symptoms_df = utils.make_df_true(symptoms, symptoms_df)
#
# pos_lst, negative_lst = utils.create_outcome_list(symptoms, symptoms_df)
# df1 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': pos_lst})
# df2 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': negative_lst})
# df1['outcome'] = 'Positive'
# df2['outcome'] = 'Negative'
# res = pd.concat([df1, df2])
# sns.barplot(x='Symptoms', y='Number of infections', data=res, hue='outcome')
# plt.show()


# symptoms = data.iloc[:, 1:-1].columns.tolist()
# symptoms_df = data.iloc[:, 1:]
# symptoms_df = utils.fetch_true_symptoms(symptoms, symptoms_df)
# print(symptoms_df)
#
# pos_lst, negative_lst = utils.create_outcome_lists(symptoms, symptoms_df)
# df1 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': pos_lst})
# df2 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': negative_lst})
# df1['outcome'] = 'Positive'
# df2['outcome'] = 'Negative'
# res = pd.concat([df1, df2])
# sns.barplot(x='Symptoms', y='Number of infections', data=res, hue='outcome', palette=constants.COLOR_PALETTE)
# plt.show()

# simple_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
# data[constants.SYSTEMIC_ILLNESS] = [simple_dict[item] for item in data[constants.SYSTEMIC_ILLNESS]]
# data_tree = data.iloc[:,:-1]
# target = data.iloc[:,-1:]
# x_train, x_test, y_train, y_test = train_test_split(data_tree, target, test_size=0.80)
# model_tree = DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=0.10)
# model_tree.fit(x_train, y_train)
# # Accuracy checken.
# y_prediction = model_tree.predict(x_test)
# print(100 * accuracy_score(y_test, y_prediction))
# text_representation = export_text(model_tree, feature_names=x_train.columns.values.tolist())
# figure = plt.figure(figsize=(10,8))
# plot_tree(model_tree, feature_names=x_train.columns, filled=True, rounded=True)
# plt.show()

simple_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
data[constants.SYSTEMIC_ILLNESS] = [simple_dict[item] for item in data[constants.SYSTEMIC_ILLNESS]]
data_tree = data.iloc[:,:-1]
target = data.iloc[:,-1:]
temp_dict = utils.create_feature_accuracy_dict(data_tree, target)
top_items = sorted(temp_dict.items(), key = lambda x: x[1], reverse=True)[:5]
pp.pprint(top_items)

# data_patient_id = data_patient_id.drop_duplicates()
# data_patient_id[constants.OCCURRENCE] = 1
# pivot = pd.pivot(data_patient_id, index=constants.PATIENT_ID, columns=constants.MONKEY_POX, values=constants.OCCURRENCE).fillna(0)
# print(pivot)
#
# frequently = apriori(pivot, min_support=0.005, use_colnames=True)

# rules = association_rules(frequently, metric='lift')
# print(frequently)
# rules.sort_values('confidence', ascending=False, inplace=True)
# print(rules)

# symptoms = data.iloc[:, 1:-1].columns.values.tolist()
# symptoms_df = data.iloc[:, 1:]
# symptoms_df = utils.clean_data(data)
# frequency = apriori(symptoms_df, min_support=0.005, use_colnames=True)
# rules = association_rules(frequency, metric='lift')
# rules.sort_values('confidence', ascending=False, inplace=True)
# rules
# symptoms = data.iloc[:, 1:-1].columns.values.tolist()
# symptoms_df = data.iloc[:, 1:]
# symptoms_df = utils.fetch_true_symptoms(symptoms, symptoms_df)
# print(symptoms_df)
#
# pos_lst, negative_lst = utils.create_outcome_lists(symptoms, symptoms_df)
# df1 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': pos_lst})
# df2 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': negative_lst})
# df1['outcome'] = 'Positive'
# df2['outcome'] = 'Negative'
# res = pd.concat([df1, df2])
# sns.barplot(x='Symptoms', y='Number of infections', data=res, hue='outcome', palette=constants.COLOR_PALETTE)
# plt.show()

# simple_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
# data_tree[constants.SYSTEMIC_ILLNESS] = [simple_dict[item] for item in data_tree[constants.SYSTEMIC_ILLNESS]]
# utils.check_over_fitting(data_tree, target)

