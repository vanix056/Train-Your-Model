import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
# Determine working directories (adjust if needed for your project structure)
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# --- (Optional) Data Reading Function ---
def read_data(dataset):
    """
    Reads CSV or Excel data given a filename.
    (Not used when uploading files via Streamlit.)
    """
    data_path = os.path.join(parent_dir, "data", dataset)
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        return df
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
        return df

# --- Data Preprocessing Function ---
def preprocess_data(df, target_col, scaler_type):
    """
    Preprocess the DataFrame by splitting into train/test,
    imputing missing values, scaling numeric columns, and one-hot encoding categorical columns.
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Process numeric features
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="mean")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        
        if scaler_type == "standard":
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
        # If "none" is selected, no scaling is performed.
    
    # Process categorical features
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        X_train_enc = pd.DataFrame(encoder.fit_transform(X_train[cat_cols]),
                                   columns=encoder.get_feature_names_out(cat_cols),
                                   index=X_train.index)
        X_test_enc = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                                  columns=encoder.get_feature_names_out(cat_cols),
                                  index=X_test.index)
        # Drop original categorical columns and concat encoded ones
        X_train = pd.concat([X_train.drop(columns=cat_cols), X_train_enc], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), X_test_enc], axis=1)
        
    return X_train, X_test, y_train, y_test

# --- Model Training Function ---
def model_train(x_train, y_train, model, model_name):
    """
    Train the model on training data and save it as a pickle file.
    """
    model.fit(x_train, y_train)
    # Ensure the trained_model directory exists
    model_dir = os.path.join(parent_dir, "trained_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model

# --- Model Evaluation Function ---
def evaluation(model, x_test, y_test):
    """
    Evaluate the trained model on the test set and return the accuracy.
    """
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return round(acc, 2)
