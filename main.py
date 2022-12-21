import pandas as pd
import numpy as np
import matplotlib
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.exceptions import DataConversionWarning
import seaborn as sns

import constants

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data = pd.read_csv('monkey_pox.csv', index_col=0)
data.rename(columns={'Systemic Illness': constants.SYSTEMIC_ILLNESS, 'Rectal Pain': constants.RECTAL_PAIN, 'Sore Throat': constants.SORE_THROAT, 'Penile Oedema': constants.PENILE_OEDEMA,
                     'Oral Lesions': constants.ORAL_LESIONS, 'Solitary Lesion': constants.SOLITARY_LESION, 'Swollen Tonsils': constants.SWOLLEN_TONSILS, 'HIV Infection': constants.HIV_INFECTION,
                     'Sexually Transmitted Infection': constants.STI }, inplace=True)
data_patient_id = pd.read_csv('monkey_pox.csv')
