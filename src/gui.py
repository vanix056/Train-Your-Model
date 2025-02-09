import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Import a wide range of models from scikit-learn (and xgboost if available)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Import utility functions
from util import preprocess_data, model_train, evaluation, tune_model

# --- Helper function for Lottie Animations ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def run():
    st.set_page_config(page_title="No Code ML Training", page_icon="🧠", layout="wide")
    # Header Animation
    lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tll0j4bb.json")
    if lottie_animation:
        st_lottie(lottie_animation, height=200)
    st.title("No Code ML Training")

    # Use tabs to organize the app
    tabs = st.tabs(["Data Upload", "Feature Engineering", "Model Training & Tuning"])

    # =====================================================
    # Tab 1: Data Upload & Preview
    # =====================================================
    with tabs[0]:
        st.header("Upload and Preview Your Data")
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state["df"] = df  # Save dataframe in session state
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("Please upload a dataset file.")

    # =====================================================
    # Tab 2: Feature Engineering
    # =====================================================
    with tabs[1]:
        st.header("Feature Engineering")
        if "df" not in st.session_state:
            st.warning("No dataset found. Please upload data in the Data Upload tab.")
        else:
            df = st.session_state["df"]
            st.subheader("Current Data Preview")
            st.dataframe(df.head())

            st.markdown("### Modify Your Data")
            # --- Option 1: Drop Columns ---
            drop_cols = st.multiselect("Select columns to drop", options=list(df.columns))
            # --- Option 2: Log Transformation ---
            num_cols = list(df.select_dtypes(include=["number"]).columns)
            log_cols = st.multiselect("Select numeric columns for log transformation", options=num_cols,
                                      help="A new column (with suffix _log) will be added. (Note: Only positive values will be transformed.)")
            # --- Option 3: Polynomial Features ---
            poly_cols = st.multiselect("Select numeric columns for polynomial features", options=num_cols,
                                       help="New polynomial features will be added for the selected columns.")
            poly_degree = st.number_input("Degree for polynomial features", min_value=2, max_value=5, value=2, step=1)

            if st.button("Apply Feature Engineering"):
                df_fe = df.copy()
                # Drop selected columns
                if drop_cols:
                    df_fe.drop(columns=drop_cols, inplace=True)
                # Apply log transformation: add new columns with suffix _log
                for col in log_cols:
                    try:
                        df_fe[f"{col}_log"] = df_fe[col].apply(lambda x: np.log(x) if (x is not None and x > 0) else np.nan)
                    except Exception as e:
                        st.error(f"Error applying log transformation on {col}: {e}")
                # Apply polynomial features for each selected column
                if poly_cols:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=int(poly_degree), include_bias=False)
                    for col in poly_cols:
                        try:
                            poly_features = poly.fit_transform(df_fe[[col]])
                            # Create new feature names
                            poly_feature_names = [f"{col}_poly_{i}" for i in range(poly_features.shape[1])]
                            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_fe.index)
                            df_fe = pd.concat([df_fe, poly_df], axis=1)
                        except Exception as e:
                            st.error(f"Error applying polynomial features on {col}: {e}")
                st.session_state["df"] = df_fe  # update dataframe in session state
                st.success("Feature engineering applied successfully!")
                st.dataframe(df_fe.head())

    # =====================================================
    # Tab 3: Model Training & Tuning
    # =====================================================
    with tabs[2]:
        st.header("Model Training & Hyperparameter Tuning")
        if "df" not in st.session_state:
            st.warning("No dataset found. Please upload data in the Data Upload tab.")
        else:
            df = st.session_state["df"]
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Select target and scaling options
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("Select the Target Column", list(df.columns))
            with col2:
                scaler_options = ["standard", "minmax", "none"]
                scaler_type = st.selectbox("Select Scaler Type for Numeric Columns", scaler_options)

            # ----------------- Extended Model List -----------------
            # Define a dictionary mapping model names to a tuple: (base_model_instance, default_parameter_grid)
            model_dict = {
                "Logistic Regression": {
                    "model": LogisticRegression(),
                    "default_grid": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"], "max_iter": [100, 200, 500]}
                },
                "Random Forest Classifier": {
                    "model": RandomForestClassifier(),
                    "default_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "criterion": ["gini", "entropy"]}
                },
                "SVC": {
                    "model": SVC(probability=True),
                    "default_grid": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]}
                },
            }
            if xgb_available:
                model_dict["XGBoost Classifier"] = {
                    "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                    "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                }
            # Add more scikit-learn models
            model_dict.update({
                "Decision Tree Classifier": {
                    "model": DecisionTreeClassifier(),
                    "default_grid": {"max_depth": [None, 5, 10, 20], "criterion": ["gini", "entropy"]}
                },
                "K-Nearest Neighbors": {
                    "model": KNeighborsClassifier(),
                    "default_grid": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
                },
                "Gradient Boosting Classifier": {
                    "model": GradientBoostingClassifier(),
                    "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                },
                "Linear Discriminant Analysis": {
                    "model": LinearDiscriminantAnalysis(),
                    "default_grid": {}
                },
                "Quadratic Discriminant Analysis": {
                    "model": QuadraticDiscriminantAnalysis(),
                    "default_grid": {}
                },
                "GaussianNB": {
                    "model": GaussianNB(),
                    "default_grid": {}
                },
                "Extra Trees Classifier": {
                    "model": ExtraTreesClassifier(),
                    "default_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30], "criterion": ["gini", "entropy"]}
                },
                "AdaBoost Classifier": {
                    "model": AdaBoostClassifier(),
                    "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]}
                },
                "Linear SVC": {
                    "model": LinearSVC(),
                    "default_grid": {"C": [0.1, 1, 10], "max_iter": [1000, 2000, 5000]}
                },
                "NuSVC": {
                    "model": NuSVC(),
                    "default_grid": {"nu": [0.1, 0.5, 0.9], "kernel": ["rbf", "linear", "poly"]}
                },
                "BernoulliNB": {
                    "model": BernoulliNB(),
                    "default_grid": {}
                }
            })

            model_name = st.text_input("Enter a name for the model", value="my_model")
            selected_model_name = st.selectbox("Select a Model", list(model_dict.keys()))
            base_model = model_dict[selected_model_name]["model"]
            default_grid = model_dict[selected_model_name]["default_grid"]

            # ------------- Hyperparameter Tuning Options -------------
            tuning_enabled = st.checkbox("Perform Hyperparameter Tuning", value=False)
            if tuning_enabled:
                st.subheader("Hyperparameter Tuning Options")
                search_method = st.radio("Select Search Method", options=["GridSearchCV", "RandomizedSearchCV"], index=0)
                cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1)
                n_iter = 10
                if search_method == "RandomizedSearchCV":
                    n_iter = st.number_input("Number of parameter settings sampled", min_value=1, max_value=100, value=10, step=1)
                # Allow the user to enter a custom parameter grid (in JSON format)
                param_grid_str = st.text_area("Parameter Grid (JSON format)", value=json.dumps(default_grid, indent=4))
                try:
                    param_grid = json.loads(param_grid_str) if param_grid_str.strip() != "" else default_grid
                except Exception as e:
                    st.error(f"Error parsing parameter grid: {e}")
                    param_grid = default_grid
            else:
                # For a few models, we allow manual hyperparameter adjustment.
                if selected_model_name == "Logistic Regression":
                    C = st.number_input("C (Inverse Regularization Strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")
                    max_iter = st.number_input("Maximum Iterations", min_value=100, max_value=10000, value=1000, step=100)
                    solver = st.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
                    base_model = LogisticRegression(C=C, max_iter=int(max_iter), solver=solver)
                elif selected_model_name == "Random Forest Classifier":
                    n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
                    max_depth = st.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
                    criterion = st.selectbox("Criterion", ["gini", "entropy"])
                    max_depth = None if max_depth == 0 else int(max_depth)
                    base_model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth, criterion=criterion)
                elif selected_model_name == "SVC":
                    C = st.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")
                    kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                    base_model = SVC(C=C, kernel=kernel, probability=True)
                # (For other models, the default parameters will be used.)

            if st.button("Train Model"):
                # Preprocess the data
                x_train, x_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
                if tuning_enabled:
                    with st.spinner("Performing hyperparameter tuning..."):
                        # Use lowercase method name for internal consistency: "grid" or "random"
                        method = "grid" if search_method == "GridSearchCV" else "random"
                        best_model, best_params = tune_model(x_train, y_train, base_model, param_grid, search_method=method, cv=cv_folds, n_iter=int(n_iter))
                    st.success(f"Best Parameters Found: {best_params}")
                    trained_model = best_model
                    # Save the tuned model
                    trained_model = model_train(x_train, y_train, trained_model, model_name)
                else:
                    trained_model = model_train(x_train, y_train, base_model, model_name)
                acc = evaluation(trained_model, x_test, y_test)
                st.success(f"Model '{model_name}' trained successfully with test accuracy: {acc}")

                # Provide a download button for the trained model
                parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(parent_dir, "trained_model", f"{model_name}.pkl")
                if os.path.exists(model_path):
                    with open(model_path, "rb") as f:
                        st.download_button("Download Trained Model",
                                           data=f.read(),
                                           file_name=f"{model_name}.pkl",
                                           mime="application/octet-stream")
                else:
                    st.error("Trained model file not found!")
                    
if __name__ == "__main__":
    run()
