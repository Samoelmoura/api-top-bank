import pickle
import pandas as pd
import numpy as np

class Top_bank(object):
    def __init__(self):
        self.cols_filtering = pickle.load(open('cols_filtering.pkl', 'rb'))
        self.ss_balance = pickle.load(open('ss_balance.pkl', 'rb'))
        self.mm_credit_score = pickle.load(open('mm_credit_score.pkl', 'rb'))
        self.mm_estimatedsalary = pickle.load(open('mm_estimatedsalary.pkl', 'rb'))
        self.map_geography = pickle.load(open('map_geography.pkl', 'rb'))
        self.map_gender = pickle.load(open('map_gender.pkl', 'rb'))
        self.map_numofproducts = pickle.load(open('map_numofproducts.pkl', 'rb'))
        self.tenure_cicle = pickle.load(open('tenure_cicle.pkl', 'rb'))
        self.cols_drop = pickle.load(open('cols_drop.pkl', 'rb'))
        self.cols_drop_split = pickle.load(open('cols_drop_split.pkl', 'rb'))
        self.map_test_id_salary = pickle.load(open('map_test_id_salary.pkl', 'rb'))
        self.model = pickle.load(open('xgb_model.pkl', 'rb'))
    
    def data_filtering(self, df):
        df.drop(self.cols_filtering, axis=1, inplace=True)
        return df
        
        
    def data_preparation(self, df):
        df['Balance'] = self.ss_balance.transform(df[['Balance']].values)
        df['CreditScore'] = self.mm_credit_score.transform(df[['CreditScore']].values)
        df['EstimatedSalary'] = self.mm_estimatedsalary.transform(df[['EstimatedSalary']].values)
        df['Age'] = np.log1p(df['Age'])
        df['Geography'] = df['Geography'].map(self.map_geography)
        df['Gender'] = df['Gender'].map(self.map_gender)
        df['NumOfProducts'] = df['NumOfProducts'].map(self.map_numofproducts)
        df['Tenure_sin'] = df['Tenure'].apply(lambda x: np.sin(x* (2*np.pi/self.tenure_cicle)))
        df['Tenure_cos'] = df['Tenure'].apply(lambda x: np.cos(x* (2*np.pi/self.tenure_cicle)))
        df.drop('Tenure', axis=1, inplace=True)
        df.drop(self.cols_drop, axis=1, inplace=True)
        df.drop(self.cols_drop_split, axis=1, inplace=True)
        return df


    def get_propensity(self, df, df_raw):
        predict_proba = self.model.predict_proba(df)
        df_raw['propensity'] = predict_proba[:, 1]
        return df_raw