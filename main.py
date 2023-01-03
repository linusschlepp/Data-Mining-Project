import pandas as pd
import numpy as np
import matplotlib
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import warnings
from sklearn import tree as tree
import graphviz as gv
from itertools import combinations
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.exceptions import DataConversionWarning
import seaborn as sns
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

import constants

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import matplotlib
from matplotlib import pyplot

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Read data from .csv file
data = pd.read_csv('monkey_pox.csv', index_col=0)
# Rename columns, substitute spaces with underscores
data.rename(columns={'Systemic Illness': constants.SYSTEMIC_ILLNESS, 'Rectal Pain': constants.RECTAL_PAIN,
                     'Sore Throat': constants.SORE_THROAT, 'Penile Oedema': constants.PENILE_OEDEMA,
                     'Oral Lesions': constants.ORAL_LESIONS, 'Solitary Lesion': constants.SOLITARY_LESION,
                     'Swollen Tonsils': constants.SWOLLEN_TONSILS, 'HIV Infection': constants.HIV_INFECTION,
                     'Sexually Transmitted Infection': constants.STI}, inplace=True)

X_data = data.copy().iloc[:, :-1]  # Dataframe, containing all features (x-values)
name_dict = {'None': 1, 'Fever': 2, 'Swollen Lymph Nodes': 3, 'Muscle Aches and Pain': 4}
# Change str-values inside Systemic Illness to corresponding int values
X_data[constants.SYSTEMIC_ILLNESS] = [name_dict[item] for item in X_data[constants.SYSTEMIC_ILLNESS]]
target = data.iloc[:, -1:] # Dataframe containing outcome (y-Values)
