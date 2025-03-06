import os
import pickle
import pandas as pd
import numpy as np
import sklearn
# Add this import at the top of utility.py
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.metrics import accuracy_score, r2_score, confusion_matrix, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set working directories (adjust as needed)
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

############################################
# Preprocessing for Classification
############################################
def preprocess_classification_data(df, target_col, scaler_type, imputer_method="mean"):
    """
    Preprocess the data for classification problems.
    Uses imputation for numeric columns and one-hot encodes categoricals.
    imputer_method: "mean", "knn", "iterative"
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if len(num_cols) > 0:
        if imputer_method=="mean":
            num_imputer = SimpleImputer(strategy="mean")
        elif imputer_method=="knn":
            num_imputer = KNNImputer()
        elif imputer_method=="iterative":
            num_imputer = IterativeImputer(random_state=42)
        else:
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
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        try:
            if tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 2):
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except Exception:
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

############################################
# Preprocessing for Regression
############################################
def preprocess_regression_data(df, target_col, scaler_type, imputer_method="median"):
    """
    Preprocess the data for regression problems.
    Uses imputation for numeric columns and one-hot encodes categoricals.
    imputer_method: "median", "knn", "iterative"
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if len(num_cols) > 0:
        if imputer_method=="median":
            num_imputer = SimpleImputer(strategy="median")
        elif imputer_method=="knn":
            num_imputer = KNNImputer()
        elif imputer_method=="iterative":
            num_imputer = IterativeImputer(random_state=42)
        else:
            num_imputer = SimpleImputer(strategy="median")
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
    
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
        try:
            if tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 2):
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            else:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        except Exception:
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

############################################
# Pipeline Creation and Model Training Function
############################################
def model_train(x_train, y_train, model, model_name, version="1.0", pipeline=False, preprocessor=None):
    """
    Train model (or pipeline if pipeline=True) and save it with version info.
    """
    if pipeline and preprocessor is not None:
        from sklearn.pipeline import Pipeline
        model_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        model_pipeline.fit(x_train, y_train)
        trained_model = model_pipeline
    else:
        model.fit(x_train, y_train)
        trained_model = model
    
    model_dir = os.path.join(parent_dir, "trained_model")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, f"{model_name}_v{version}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(trained_model, f)
    return trained_model

############################################
# Model Evaluation Function with Extra Metrics
############################################
def evaluation(model, x_test, y_test, problem_type="Classification"):
    """
    Evaluate the model and return a dictionary of metrics.
    For classification: accuracy, confusion matrix, precision, recall, F1-score.
    For regression: R², MAE, RMSE.
    """
    y_pred = model.predict(x_test)
    
    if problem_type == "Classification":
        from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        return {
            "accuracy": round(acc, 4),
            "confusion_matrix": cm,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }
    else:
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mse = mean_squared_error(y_test, y_pred)
        
        return {
            "r2": round(r2, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mse": round(mse, 4)
        }
############################################
# Hyperparameter Tuning Function (including Optuna)
############################################
def tune_model(x_train, y_train, model, param_grid, search_method="grid", cv=5, n_iter=10):
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    try:
        from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV
    except ImportError:
        HalvingGridSearchCV = None
        HalvingRandomSearchCV = None

    if search_method == "grid":
        search = GridSearchCV(model, param_grid, cv=cv)
    elif search_method == "random":
        search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter)
    elif search_method == "halving_grid" and HalvingGridSearchCV is not None:
        search = HalvingGridSearchCV(model, param_grid, cv=cv)
    elif search_method == "halving_random" and HalvingRandomSearchCV is not None:
        search = HalvingRandomSearchCV(model, param_grid, cv=cv, n_candidates=n_iter)
    elif search_method == "optuna":
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is not installed.")
        def objective(trial):
            params = {}
            for key, values in param_grid.items():
                params[key] = trial.suggest_categorical(key, values)
            model.set_params(**params)
            from sklearn.model_selection import cross_val_score
            score = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy" if hasattr(model, "predict_proba") else "r2").mean()
            return score
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_iter)
        best_params = study.best_params
        model.set_params(**best_params)
        model.fit(x_train, y_train)
        return model, best_params
    else:
        search = GridSearchCV(model, param_grid, cv=cv)
    search.fit(x_train, y_train)
    return search.best_estimator_, search.best_params_

############################################
# SHAP Explanation Function
############################################
def compute_shap(model, x_sample):
    try:
        import shap
    except ImportError:
        return None, "SHAP not installed"
    explainer = shap.Explainer(model.predict, x_sample)
    shap_values = explainer(x_sample)
    return shap_values, None
