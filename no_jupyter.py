import utils
from main import *
import constants

test_dict = dict.fromkeys(data[constants.SYSTEMIC_ILLNESS].unique())

for index in data.index:
    if data['MonkeyPox'][index] == 'Positive':
        test_dict[data[constants.SYSTEMIC_ILLNESS][index]] = int(
            test_dict[data[constants.SYSTEMIC_ILLNESS][index]] or 0) + 1

print(test_dict)
df = pd.DataFrame(test_dict.items())

df.rename(columns={0: 'illness', 1: 'quantity'}, inplace=True)
plt.bar(df['illness'], df['quantity'])
plt.title('Cases of monkeypox seperated by systemic illness')
plt.xlabel('Illness')
plt.ylabel('Occurrence')
plt.show()
simple_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
data[constants.SYSTEMIC_ILLNESS] = [simple_dict[item] for item in data[constants.SYSTEMIC_ILLNESS]]
data_tree = data.iloc[:, :-1]
target = data.iloc[:, -1:]
ran_stream = 23
x_train, x_test, y_train, y_test = train_test_split(data_tree, target, random_state=ran_stream)

model = GaussianNB()
# x_train.loc[x_train['MonkeyPox'] == 'Negative'] = False
# x_train.loc[x_train['MonkeyPox'] == 'Positive'] = True
# x_train['MonkeyPox'] = x_train['MonkeyPox'].astype(int)
# model.fit(y_train, x_train)
model.fit(x_train, y_train)
y_prediction = model.predict(x_test)
print(100 * accuracy_score(y_test, y_prediction))

# model_tree = DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=5)
# model_tree.fit(data_tree, target)
# text_representation = export_text(model_tree, feature_names=data_tree.columns.values.tolist())
# figure = plt.figure(figsize=(10,8))
# plot_tree(model_tree, feature_names=data_tree.columns, filled=True, rounded=True)
# plt.show()


df = data.groupby(by='MonkeyPox').sum()
symptoms = data.iloc[:, 1:-1].columns.values.tolist()
# Have to be true to be counted
pos_lst, negative_lst = utils.create_outcome_list(symptoms, df)
df1 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': pos_lst})
df2 = pd.DataFrame({'Symptoms': symptoms, 'Number of infections': negative_lst})
df1['outcome'] = 'Positive'
df2['outcome'] = 'Negative'
res = pd.concat([df1, df2])
sns.barplot(x='Symptoms', y='Number of infections', data=res, hue='outcome')
plt.show()
plt.show()
