#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 12:03:34 2023

@author: isabella
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from time import time
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def main():
    
    st.title("Plots for Credit Card Acceptance")


    df = pd.read_csv("https://github.com/povembu/D590-ADS-Project/blob/main/Datasets/cc_processed.csv?raw=true")

    st.write("Data overview:")
    st.write(df.head())



    graphdata = ['Annual_income', 'Employed_days', 'label']
    new_df = df[graphdata].copy()
    new_df['Employed_days'] = new_df['Employed_days'].abs()

    fig = px.scatter(new_df, x='Annual_income', y='Employed_days', color='label', hover_data=['Annual_income', 'Employed_days', 'label'])

    fig.update_layout(
    title='Credit Card Approval Based on Factors Employment and Income',
    xaxis_title='Annual Income',
    yaxis_title='Days of Employment'
    )


    st.title('Interactive Scatter Plot Graph')


    st.plotly_chart(fig)
   
    

    #Feature importance plot
    cc_df    = pd.read_csv("https://github.com/povembu/D590-ADS-Project/blob/main/Datasets/Credit_card.csv?raw=true")
    cc_label = pd.read_csv("https://github.com/povembu/D590-ADS-Project/blob/main/Datasets/Credit_card_label.csv?raw=true")
    cc_final = pd.merge(cc_df, cc_label, on='Ind_ID')
    pre_cc = cc_final
    pre_cc['Annual_income'].fillna(pre_cc['Annual_income'].median(),inplace =True)
    pre_cc['Birthday_count'].fillna(pre_cc['Birthday_count'].mean(),inplace =True)
   
    pre_cc['GENDER'].fillna(pre_cc['GENDER'].mode()[0],inplace =True)
    pre_cc.dropna(subset=['Type_Occupation'], inplace=True)
    pre_cc['Age_conv'] = np.abs(pre_cc['Birthday_count'])/365
    pre_cc['Employed_years'] = (pre_cc['Employed_days']/365)
   
    ohe = OneHotEncoder(sparse=False,handle_unknown='ignore')

    cat_attribs = ['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']

    enc_df = pd.DataFrame(ohe.fit_transform(pre_cc[['GENDER','Car_Owner','Propert_Owner','Type_Income','EDUCATION','Marital_status','Housing_type','Type_Occupation']]), columns = ohe.get_feature_names_out())
    cc_df = df.join(enc_df)
    cc_df.drop(cat_attribs,axis=1,inplace=True)
   
   
    X = cc_df.drop(['Ind_ID','label'],axis = 'columns')
    y = cc_df['label']
   
    oversample = SMOTE()

    X, y = oversample.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,stratify = y,random_state = 42)
 

    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
   
    X_train = X_train_scaled
    X_test = X_test_scaled
   
    np.random.seed(42)
    import json

    rfc = RandomForestClassifier(class_weight='balanced', random_state=42)

    params_grid = {
            'max_depth': [5,10,15],
            'max_features': [10,15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1,3],
            'bootstrap': [True],
            'n_estimators':[25,50,100]
          }


    rfc_grid = GridSearchCV(rfc, params_grid,scoring='f1',cv =5, n_jobs = -1)
    rfc_grid.fit(X_train,y_train)
   
    # ("Best parameters: ")
    best_params = rfc_grid.best_estimator_.get_params()
    # print(best_params)
    param_dump = []
    for i in sorted(params_grid):
        param_dump.append((i, best_params[i]))
        # ("\t"+str(i)+": " +str(best_params[i]))

    # start = time()
    rfc_model = rfc_grid.best_estimator_.fit(X_train,y_train)
   
    #roc-auc score
    best_train_auc = roc_auc_score(y_train, rfc_model.predict_proba(X_train)[:, 1])
    # train_time = round(time()- start, 4)
    best_test_auc = roc_auc_score(y_test, rfc_model.predict_proba(X_test)[:, 1])
   
    #f1 score
    best_train_f1= f1_score(y_train, rfc_model.predict(X_train))
    best_test_f1 = f1_score(y_test, rfc_model.predict(X_test))
   
    #precision
    best_train_precision = precision_score(y_train, rfc_model.predict(X_train))
    best_test_precision  = precision_score(y_test, rfc_model.predict(X_test))
   
    #recall
    best_train_recall = recall_score(y_train, rfc_model.predict(X_train))
    best_test_recall  = recall_score(y_test, rfc_model.predict(X_test))
   
    #confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, rfc_model.predict(X_test)).ravel()
   
   

    feature_importances = rfc_model.feature_importances_
    feature_names = X.columns
   
   
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

   
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    st.title('Random Forest Regression Feature Importance')

    st.write('Feature Importances:')
    st.write(feature_importance_df)

    fig = px.bar(feature_importance_df, x='Feature', y='Importance', title='Feature Importances')
    st.plotly_chart(fig)
   
main()
