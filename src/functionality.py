import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
from streamlit_lottie import st_lottie

# Import models for classification and regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, BernoulliNB

try:
    from xgboost import XGBClassifier, XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# Import our helper functions from util.py:
from util import (
    preprocess_classification_data,
    preprocess_regression_data,
    model_train,
    evaluation,
    tune_model
)

############################################
# Helper Function for Lottie Animations
############################################
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

############################################
# Main GUI Function
############################################
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
            st.info("Please upload a dataset file to begin.")
    
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
    
            st.markdown("### Basic Feature Engineering Options")
            # Option 1: Drop Columns
            drop_cols = st.multiselect("Select columns to drop", options=list(df.columns))
            # Option 2: Log Transformation for numeric columns
            num_cols = list(df.select_dtypes(include=["number"]).columns)
            log_cols = st.multiselect("Select numeric columns for log transformation", options=num_cols,
                                      help="A new column (with suffix _log) will be added. (Only positive values are transformed.)")
            # Option 3: Polynomial Features
            poly_cols = st.multiselect("Select numeric columns for polynomial features", options=num_cols,
                                       help="New polynomial features will be added for the selected columns.")
            poly_degree = st.number_input("Degree for polynomial features", min_value=2, max_value=5, value=2, step=1)
    
            st.markdown("### Additional Feature Engineering Options")
            # Option 1: Binning/Discretization
            binning_enabled = st.checkbox("Apply Binning/Discretization on numeric columns")
            if binning_enabled:
                bin_cols = st.multiselect("Select numeric columns to bin", options=num_cols)
                n_bins = st.number_input("Number of bins", min_value=2, max_value=20, value=5, step=1)
                bin_method = st.selectbox("Binning method", ["Equal Width", "Equal Frequency"])
            # Option 2: Interaction Features
            interaction_enabled = st.checkbox("Create Interaction Features for numeric columns")
            if interaction_enabled:
                interaction_cols = st.multiselect("Select numeric columns to interact", options=num_cols)
            # Option 3: Square Root Transformation
            sqrt_enabled = st.checkbox("Apply Square Root Transformation to numeric columns")
            if sqrt_enabled:
                sqrt_cols = st.multiselect("Select numeric columns for square root transformation", options=num_cols)
            # Option 4: Exponential Transformation
            exp_enabled = st.checkbox("Apply Exponential Transformation to numeric columns")
            if exp_enabled:
                exp_cols = st.multiselect("Select numeric columns for exponential transformation", options=num_cols)
            # Option 5: Power Transformation (Yeo-Johnson)
            power_enabled = st.checkbox("Apply Power Transformation (Yeo-Johnson) to numeric columns")
            if power_enabled:
                power_cols = st.multiselect("Select numeric columns for power transformation", options=num_cols)
            # Option 6: Outlier Removal
            outlier_enabled = st.checkbox("Remove Outliers based on IQR for numeric columns")
            if outlier_enabled:
                outlier_cols = st.multiselect("Select numeric columns for outlier removal", options=num_cols)
                iqr_multiplier = st.number_input("IQR multiplier", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
            # Option 7: Feature Crossing for Categorical Columns
            cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
            cross_enabled = st.checkbox("Create Feature Cross for categorical columns")
            if cross_enabled:
                cross_cols = st.multiselect("Select two categorical columns for feature crossing", options=cat_cols, max_selections=2)
            # Option 8: Frequency Encoding for Categorical Columns
            freq_enabled = st.checkbox("Apply Frequency Encoding to categorical columns")
            if freq_enabled:
                freq_cols = st.multiselect("Select categorical columns for frequency encoding", options=cat_cols)
            # Option 9: Target Encoding for Categorical Columns
            target_encode_enabled = st.checkbox("Apply Target Encoding to categorical columns (requires target column)")
            if target_encode_enabled:
                target_encode_cols = st.multiselect("Select categorical columns for target encoding", options=cat_cols)
            # Option 10: Datetime Feature Extraction
            datetime_enabled = st.checkbox("Extract DateTime features")
            if datetime_enabled:
                datetime_cols = st.multiselect("Select datetime columns", options=list(df.columns))
            # Option 11: Text Length Feature for Text Columns
            text_len_enabled = st.checkbox("Create Text Length Feature for text columns")
            if text_len_enabled:
                text_cols = st.multiselect("Select text columns", options=list(df.select_dtypes(include=["object"]).columns))
            # Option 12: Replace Missing Values with a Constant
            replace_missing_enabled = st.checkbox("Replace Missing Values with a Constant")
            if replace_missing_enabled:
                replace_missing_cols = st.multiselect("Select columns to replace missing values", options=list(df.columns))
                constant_value = st.text_input("Constant value to fill missing values", value="0")
            # Option 13: Add Missing Indicator for Columns
            missing_indicator_enabled = st.checkbox("Add Missing Value Indicator for columns")
            if missing_indicator_enabled:
                missing_indicator_cols = st.multiselect("Select columns to add missing indicator", options=list(df.columns))
            # Option 14: Robust Scaling for Numeric Columns
            robust_scaling_enabled = st.checkbox("Apply Robust Scaling to numeric columns")
            if robust_scaling_enabled:
                robust_scaling_cols = st.multiselect("Select numeric columns for robust scaling", options=num_cols)
            # Option 15: Rank Transformation for Numeric Columns
            rank_enabled = st.checkbox("Apply Rank Transformation to numeric columns")
            if rank_enabled:
                rank_cols = st.multiselect("Select numeric columns for rank transformation", options=num_cols)
            # Option 16: Log1p Transformation for Numeric Columns
            log1p_enabled = st.checkbox("Apply Log1p Transformation to numeric columns")
            if log1p_enabled:
                log1p_cols = st.multiselect("Select numeric columns for log1p transformation", options=num_cols)
            # Option 17: Interaction Only Polynomial Features
            poly_interaction_enabled = st.checkbox("Apply Interaction Only Polynomial Features")
            if poly_interaction_enabled:
                poly_interaction_cols = st.multiselect("Select numeric columns for interaction-only polynomial features", options=num_cols)
            
            if st.button("Apply Feature Engineering"):
                df_fe = df.copy()
                # Basic Feature Engineering
                if drop_cols:
                    df_fe.drop(columns=drop_cols, inplace=True)
                for col in log_cols:
                    try:
                        df_fe[f"{col}_log"] = df_fe[col].apply(lambda x: np.log(x) if (x is not None and x > 0) else np.nan)
                    except Exception as e:
                        st.error(f"Error applying log transformation on {col}: {e}")
                if poly_cols:
                    from sklearn.preprocessing import PolynomialFeatures
                    poly = PolynomialFeatures(degree=int(poly_degree), include_bias=False)
                    for col in poly_cols:
                        try:
                            poly_features = poly.fit_transform(df_fe[[col]])
                            poly_feature_names = [f"{col}_poly_{i}" for i in range(poly_features.shape[1])]
                            poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_fe.index)
                            df_fe = pd.concat([df_fe, poly_df], axis=1)
                        except Exception as e:
                            st.error(f"Error applying polynomial features on {col}: {e}")
                # Additional Feature Engineering Options
                if binning_enabled and bin_cols:
                    for col in bin_cols:
                        try:
                            if bin_method == "Equal Width":
                                df_fe[f"{col}_binned"] = pd.cut(df_fe[col], bins=n_bins, labels=False)
                            else:
                                df_fe[f"{col}_binned"] = pd.qcut(df_fe[col], q=n_bins, labels=False, duplicates='drop')
                        except Exception as e:
                            st.error(f"Error applying binning on {col}: {e}")
                if interaction_enabled and len(interaction_cols) > 1:
                    try:
                        df_fe["interaction_" + "_".join(interaction_cols)] = df_fe[interaction_cols].prod(axis=1)
                    except Exception as e:
                        st.error(f"Error creating interaction features: {e}")
                if sqrt_enabled and sqrt_cols:
                    for col in sqrt_cols:
                        try:
                            df_fe[f"{col}_sqrt"] = df_fe[col].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
                        except Exception as e:
                            st.error(f"Error applying square root transformation on {col}: {e}")
                if exp_enabled and exp_cols:
                    for col in exp_cols:
                        try:
                            df_fe[f"{col}_exp"] = np.exp(df_fe[col])
                        except Exception as e:
                            st.error(f"Error applying exponential transformation on {col}: {e}")
                if power_enabled and power_cols:
                    try:
                        from sklearn.preprocessing import PowerTransformer
                        pt = PowerTransformer(method="yeo-johnson")
                        df_power = pd.DataFrame(pt.fit_transform(df_fe[power_cols]), columns=[f"{col}_power" for col in power_cols], index=df_fe.index)
                        df_fe = pd.concat([df_fe, df_power], axis=1)
                    except Exception as e:
                        st.error(f"Error applying power transformation: {e}")
                if outlier_enabled and outlier_cols:
                    try:
                        for col in outlier_cols:
                            Q1 = df_fe[col].quantile(0.25)
                            Q3 = df_fe[col].quantile(0.75)
                            IQR = Q3 - Q1
                            df_fe = df_fe[(df_fe[col] >= Q1 - iqr_multiplier * IQR) & (df_fe[col] <= Q3 + iqr_multiplier * IQR)]
                    except Exception as e:
                        st.error(f"Error removing outliers: {e}")
                if cross_enabled and len(cross_cols) == 2:
                    try:
                        df_fe[f"cross_{cross_cols[0]}_{cross_cols[1]}"] = df_fe[cross_cols[0]].astype(str) + "_" + df_fe[cross_cols[1]].astype(str)
                    except Exception as e:
                        st.error(f"Error creating feature cross: {e}")
                if freq_enabled and freq_cols:
                    for col in freq_cols:
                        try:
                            freq = df_fe[col].value_counts()
                            df_fe[f"{col}_freq"] = df_fe[col].map(freq)
                        except Exception as e:
                            st.error(f"Error applying frequency encoding on {col}: {e}")
                if target_encode_enabled and target_encode_cols and 'target_column' in st.session_state:
                    for col in target_encode_cols:
                        try:
                            mapping = df_fe.groupby(col)[st.session_state["target_column"]].mean()
                            df_fe[f"{col}_target_enc"] = df_fe[col].map(mapping)
                        except Exception as e:
                            st.error(f"Error applying target encoding on {col}: {e}")
                if datetime_enabled and datetime_cols:
                    for col in datetime_cols:
                        try:
                            df_fe[col] = pd.to_datetime(df_fe[col], errors='coerce')
                            df_fe[f"{col}_year"] = df_fe[col].dt.year
                            df_fe[f"{col}_month"] = df_fe[col].dt.month
                            df_fe[f"{col}_day"] = df_fe[col].dt.day
                            df_fe[f"{col}_weekday"] = df_fe[col].dt.weekday
                        except Exception as e:
                            st.error(f"Error extracting datetime features from {col}: {e}")
                if text_len_enabled and text_cols:
                    for col in text_cols:
                        try:
                            df_fe[f"{col}_len"] = df_fe[col].astype(str).apply(len)
                        except Exception as e:
                            st.error(f"Error creating text length feature for {col}: {e}")
                if replace_missing_enabled and replace_missing_cols:
                    for col in replace_missing_cols:
                        try:
                            df_fe[col] = df_fe[col].fillna(constant_value)
                        except Exception as e:
                            st.error(f"Error replacing missing values in {col}: {e}")
                if missing_indicator_enabled and missing_indicator_cols:
                    for col in missing_indicator_cols:
                        try:
                            df_fe[f"{col}_missing"] = df_fe[col].isna().astype(int)
                        except Exception as e:
                            st.error(f"Error adding missing indicator for {col}: {e}")
                if robust_scaling_enabled and robust_scaling_cols:
                    try:
                        from sklearn.preprocessing import RobustScaler
                        rs = RobustScaler()
                        df_rs = pd.DataFrame(rs.fit_transform(df_fe[robust_scaling_cols]), columns=[f"{col}_robust" for col in robust_scaling_cols], index=df_fe.index)
                        df_fe = pd.concat([df_fe, df_rs], axis=1)
                    except Exception as e:
                        st.error(f"Error applying robust scaling: {e}")
                if rank_enabled and rank_cols:
                    for col in rank_cols:
                        try:
                            df_fe[f"{col}_rank"] = df_fe[col].rank()
                        except Exception as e:
                            st.error(f"Error applying rank transformation on {col}: {e}")
                if log1p_enabled and log1p_cols:
                    for col in log1p_cols:
                        try:
                            df_fe[f"{col}_log1p"] = np.log1p(df_fe[col])
                        except Exception as e:
                            st.error(f"Error applying log1p transformation on {col}: {e}")
                if poly_interaction_enabled and poly_interaction_cols:
                    try:
                        from sklearn.preprocessing import PolynomialFeatures
                        poly_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                        poly_int_features = poly_int.fit_transform(df_fe[poly_interaction_cols])
                        poly_int_feature_names = poly_int.get_feature_names_out(poly_interaction_cols)
                        poly_int_df = pd.DataFrame(poly_int_features, columns=[f"{name}_int" for name in poly_int_feature_names], index=df_fe.index)
                        df_fe = pd.concat([df_fe, poly_int_df], axis=1)
                    except Exception as e:
                        st.error(f"Error applying interaction-only polynomial features: {e}")
    
                st.session_state["df"] = df_fe
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
            
            # Let the user select the problem type
            problem_type = st.radio("Select Problem Type", options=["Classification", "Regression"])
            
            col1, col2 = st.columns(2)
            with col1:
                target_column = st.selectbox("Select the Target Column", list(df.columns))
                st.session_state["target_column"] = target_column
            with col2:
                scaler_options = ["standard", "minmax", "none"]
                scaler_type = st.selectbox("Select Scaler Type for Numeric Columns", scaler_options)
            
            # ----------------- Model List -----------------
            if problem_type == "Classification":
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
            else:
                # Regression models
                model_dict = {
                    "Linear Regression": {
                        "model": LinearRegression(),
                        "default_grid": {"fit_intercept": [True, False]}
                    },
                    "Random Forest Regressor": {
                        "model": RandomForestRegressor(),
                        "default_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}
                    },
                    "Decision Tree Regressor": {
                        "model": DecisionTreeRegressor(),
                        "default_grid": {"max_depth": [None, 5, 10, 20]}
                    },
                    "K-Nearest Neighbors Regressor": {
                        "model": KNeighborsRegressor(),
                        "default_grid": {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
                    },
                    "Gradient Boosting Regressor": {
                        "model": GradientBoostingRegressor(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                    },
                }
                if xgb_available:
                    model_dict["XGBoost Regressor"] = {
                        "model": XGBRegressor(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                    }
    
            model_name = st.text_input("Enter a name for the model", value="my_model")
            selected_model_name = st.selectbox("Select a Model", list(model_dict.keys()))
            base_model = model_dict[selected_model_name]["model"]
            default_grid = model_dict[selected_model_name]["default_grid"]
    
            # --- Hyperparameter Tuning Options ---
            tuning_enabled = st.checkbox("Perform Hyperparameter Tuning", value=False)
            if tuning_enabled:
                st.subheader("Hyperparameter Tuning Options")
                search_method_choice = st.radio("Select Search Method", options=["GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV"], index=0)
                cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1)
                n_iter = 10
                if search_method_choice in ["RandomizedSearchCV", "HalvingRandomSearchCV"]:
                    n_iter = st.number_input("Number of parameter settings sampled", min_value=1, max_value=100, value=10, step=1)
                param_grid_str = st.text_area("Parameter Grid (JSON format)", value=json.dumps(default_grid, indent=4))
                try:
                    param_grid = json.loads(param_grid_str) if param_grid_str.strip() != "" else default_grid
                except Exception as e:
                    st.error(f"Error parsing parameter grid: {e}")
                    param_grid = default_grid
            else:
                # Optional manual parameter tweaking for select models
                if problem_type == "Classification":
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
                else:
                    if selected_model_name == "Linear Regression":
                        fit_intercept = st.selectbox("Fit Intercept", [True, False])
                        base_model = LinearRegression(fit_intercept=fit_intercept)
                    elif selected_model_name == "Random Forest Regressor":
                        n_estimators = st.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
                        max_depth = st.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
                        max_depth = None if max_depth == 0 else int(max_depth)
                        base_model = RandomForestRegressor(n_estimators=int(n_estimators), max_depth=max_depth)
                    elif selected_model_name == "Decision Tree Regressor":
                        max_depth = st.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
                        max_depth = None if max_depth == 0 else int(max_depth)
                        base_model = DecisionTreeRegressor(max_depth=max_depth)
                    elif selected_model_name == "K-Nearest Neighbors Regressor":
                        n_neighbors = st.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, step=1)
                        weights = st.selectbox("Weights", ["uniform", "distance"])
                        base_model = KNeighborsRegressor(n_neighbors=int(n_neighbors), weights=weights)
                    elif selected_model_name == "Gradient Boosting Regressor":
                        n_estimators = st.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
                        learning_rate = st.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
                        max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
                        base_model = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth))
    
            if st.button("Train Model"):
                # Call the appropriate preprocessing function based on problem type.
                if problem_type == "Classification":
                    x_train, x_test, y_train, y_test = preprocess_classification_data(df, target_column, scaler_type)
                else:
                    x_train, x_test, y_train, y_test = preprocess_regression_data(df, target_column, scaler_type)
    
                # If hyperparameter tuning is enabled
                if tuning_enabled:
                    with st.spinner("Performing hyperparameter tuning..."):
                        if search_method_choice == "GridSearchCV":
                            method = "grid"
                        elif search_method_choice == "RandomizedSearchCV":
                            method = "random"
                        elif search_method_choice == "HalvingGridSearchCV":
                            method = "halving_grid"
                        elif search_method_choice == "HalvingRandomSearchCV":
                            method = "halving_random"
                        best_model, best_params = tune_model(x_train, y_train, base_model, param_grid, search_method=method, cv=cv_folds, n_iter=int(n_iter))
                    st.success(f"Best Parameters Found: {best_params}")
                    trained_model = best_model
                    trained_model = model_train(x_train, y_train, trained_model, model_name)
                else:
                    trained_model = model_train(x_train, y_train, base_model, model_name)
    
                # Evaluate model using the appropriate metric
                metric = evaluation(trained_model, x_test, y_test, problem_type)
                if problem_type == "Classification":
                    st.success(f"Model '{model_name}' trained successfully with test accuracy: {metric}")
                else:
                    st.success(f"Model '{model_name}' trained successfully with R² score: {metric}")
    
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
