import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score
import warnings
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data = pd.read_csv('monkey_pox.csv', index_col=0)
