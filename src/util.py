import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

working_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(working_dir)


def read_data(dataset):
    data_path = f'{parent_dir}/data/{dataset}'
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        return df
    elif data_path.endswith('.xlsx') or data_path.endswith('xls'):
        df = pd.read_excel(data_path)
        return df
        
    
def preprocess_data():
    
def model_train():
    
def evaluation():

