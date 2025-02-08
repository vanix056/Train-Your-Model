import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def read_data(filename):
    data_path = f'{parent_dir}/data/{filename}'
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        return pd.read_excel(data_path)
    return None

def read_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    return None

def preprocess_data(df, target_col, scaler_type='None'):
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify column types
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler() if scaler_type == 'Standard Scaler' 
                  else MinMaxScaler() if scaler_type == 'MinMax Scaler' 
                  else 'passthrough')
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

def save_model(pipeline, model_name):
    os.makedirs(f'{parent_dir}/trained_models', exist_ok=True)
    with open(f'{parent_dir}/trained_models/{model_name}.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }