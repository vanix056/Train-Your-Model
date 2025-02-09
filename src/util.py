import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Set working directories (adjust as needed)
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def preprocess_data(df, target_col, scaler_type):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
        # If "none", no scaling is applied.
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        # Use the appropriate parameter name based on scikit-learn version
        try:
            if tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 2):
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except Exception as e:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        
        X_train_enc = pd.DataFrame(
            encoder.fit_transform(X_train[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_train.index
        )
        X_test_enc = pd.DataFrame(
            encoder.transform(X_test[cat_cols]),
            columns=encoder.get_feature_names_out(cat_cols),
            index=X_test.index
        )
        X_train = pd.concat([X_train.drop(columns=cat_cols), X_train_enc], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), X_test_enc], axis=1)
        
    return X_train, X_test, y_train, y_test
def model_train(x_train, y_train, model, model_name):
    """
    Trains the model on the training data and saves it as a pickle file.
    """
    model.fit(x_train, y_train)
    model_dir = os.path.join(parent_dir, "trained_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model

def evaluation(model, x_test, y_test):
    """
    Evaluates the model on the test set and returns the accuracy.
    """
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    return round(acc, 2)

def tune_model(x_train, y_train, model, param_grid, search_method="grid", cv=5, n_iter=10):
    """
    Tunes hyperparameters using GridSearchCV or RandomizedSearchCV.
    
    Parameters:
      - x_train, y_train: Training data.
      - model: Base model.
      - param_grid: Dictionary of parameters to search.
      - search_method: "grid" or "random".
      - cv: Number of cross-validation folds.
      - n_iter: Number of iterations for RandomizedSearchCV.
    
    Returns:
      - best_estimator: The model with the best found parameters.
      - best_params: Dictionary of the best parameters.
    """
    if search_method == "grid":
        from sklearn.model_selection import GridSearchCV
        search = GridSearchCV(model, param_grid, cv=cv)
    else:
        from sklearn.model_selection import RandomizedSearchCV
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_
