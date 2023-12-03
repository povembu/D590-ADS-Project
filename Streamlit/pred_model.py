import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from zipfile import ZipFile
import textwrap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,f1_score, precision_score, roc_auc_score, recall_score,roc_curve
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import pickle




#model
data_url = "https://github.com/povembu/D590-ADS-Project/blob/main/Datasets/cc_processed.csv?raw=true"

def load_data():
        data = pd.read_csv(data_url)
        return data

pre_cc = load_data()

#preprocess

X = pre_cc.drop(['Ind_ID','label'],axis = 'columns')
y = pre_cc['label']

# oversample = SMOTE()

# X, y = oversample.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 42)

#ohe inside pipeline so that ohe can also be applied on user data from the app
model_pipe = Pipeline([
        ("ohe",OneHotEncoder(sparse=False,handle_unknown='ignore')),
        ("rfc",RandomForestClassifier(max_depth=15,max_features=10,min_samples_leaf=1,min_samples_split=2,n_estimators=100,bootstrap=True,class_weight='balanced', random_state=42))
])

fit_model = model_pipe.fit(X_train,y_train)

with open('rfc_model.pkl','wb') as fid:
        pickle.dump(fit_model,fid)

