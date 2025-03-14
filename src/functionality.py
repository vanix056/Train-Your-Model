from sklearn.experimental import enable_iterative_imputer
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_lottie import st_lottie
import shap
import pickle


# Import models for classification and regression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.impute import IterativeImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    StackingClassifier,
    StackingRegressor
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

# For text feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# For dimensionality reduction
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE

try:
    import umap
    umap_available = True
except ImportError:
    umap_available = False

# Import our helper functions from util.py:
from util import (
    preprocess_classification_data,
    preprocess_regression_data,
    model_train,
    evaluation,
    tune_model,
    compute_shap
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
# Sanatizing Column Names
############################################
def sanitize_column_names(df):
    """Replace invalid characters in column names with underscores."""
    df.columns = [
        col.replace("[", "_").replace("]", "_").replace("<", "_").replace(">", "_")
        for col in df.columns
    ]
    return df

############################################
# Caching Data Loading
############################################
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df = sanitize_column_names(df)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

############################################
# Main GUI Function
############################################
def run():
    st.set_page_config(page_title="No Code ML Training", page_icon="🧠", layout="wide")
     # Initialize session state variables
    required_keys = {
        "df": pd.DataFrame(),
        "target_column": None,
        "trained_model": None,
        "x_test": None,
        "problem_type": None,
        "feature_engineered": False
    }
    
    for key, default in required_keys.items():
        if key not in st.session_state:
            st.session_state[key] = default
            
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
        if uploaded_file and not st.session_state.feature_engineered:
            try:
                df = load_data(uploaded_file)
                if df is not None and not df.empty:
                    df = sanitize_column_names(df)
                    st.session_state.df = df
                    st.dataframe(df.head())
            except Exception as e:
                st.error(f"Load error: {str(e)}")        
        else:
            st.info("Please upload a dataset file to begin.")
    
    # =====================================================
    # Tab 2: Feature Engineering
    # =====================================================
    with tabs[1]:
        st.header("Feature Engineering")
        if st.session_state.df is None:
            st.warning("Upload data first")
        
        else:
            df = st.session_state["df"]
            st.subheader("Current Data Preview")
            st.dataframe(df.head())
    
            st.markdown("### Basic Feature Engineering Options")
            drop_cols = st.multiselect("Select columns to drop", options=list(df.columns))
            num_cols = list(df.select_dtypes(include=["number"]).columns)
            log_cols = st.multiselect("Select numeric columns for log transformation", options=num_cols,
                                      help="Creates new columns with suffix _log (only for positive values)")
            poly_cols = st.multiselect("Select numeric columns for polynomial features", options=num_cols)
            poly_degree = st.number_input("Degree for polynomial features", min_value=2, max_value=5, value=2, step=1)
    
            st.markdown("### Additional Feature Engineering Options")
            binning_enabled = st.checkbox("Apply Binning/Discretization on numeric columns")
            if binning_enabled:
                bin_cols = st.multiselect("Select numeric columns to bin", options=num_cols)
                n_bins = st.number_input("Number of bins", min_value=2, max_value=20, value=5, step=1)
                bin_method = st.selectbox("Binning method", ["Equal Width", "Equal Frequency"])
            interaction_enabled = st.checkbox("Create Interaction Features for numeric columns (multiplication)")
            if interaction_enabled:
                interaction_cols = st.multiselect("Select numeric columns to interact", options=num_cols)
            sqrt_enabled = st.checkbox("Apply Square Root Transformation")
            if sqrt_enabled:
                sqrt_cols = st.multiselect("Select numeric columns for square root transformation", options=num_cols)
            exp_enabled = st.checkbox("Apply Exponential Transformation")
            if exp_enabled:
                exp_cols = st.multiselect("Select numeric columns for exponential transformation", options=num_cols)
            power_enabled = st.checkbox("Apply Power Transformation (Yeo-Johnson)")
            if power_enabled:
                power_cols = st.multiselect("Select numeric columns for power transformation", options=num_cols)
            outlier_enabled = st.checkbox("Remove Outliers based on IQR")
            if outlier_enabled:
                outlier_cols = st.multiselect("Select numeric columns for outlier removal", options=num_cols)
                iqr_multiplier = st.number_input("IQR multiplier", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
            cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
            cross_enabled = st.checkbox("Create Feature Cross for categorical columns")
            if cross_enabled:
                cross_cols = st.multiselect("Select two categorical columns for feature crossing", options=cat_cols, max_selections=2)
            freq_enabled = st.checkbox("Apply Frequency Encoding to categorical columns")
            if freq_enabled:
                freq_cols = st.multiselect("Select categorical columns for frequency encoding", options=cat_cols)
            target_encode_enabled = st.checkbox("Apply Target Encoding to categorical columns (requires target column)")
            if target_encode_enabled:
                target_encode_cols = st.multiselect("Select categorical columns for target encoding", options=cat_cols)
            datetime_enabled = st.checkbox("Extract DateTime features")
            if datetime_enabled:
                datetime_cols = st.multiselect("Select datetime columns", options=list(df.columns))
            text_len_enabled = st.checkbox("Create Text Length Feature for text columns")
            if text_len_enabled:
                text_cols = st.multiselect("Select text columns", options=list(df.select_dtypes(include=["object"]).columns))
            replace_missing_enabled = st.checkbox("Replace Missing Values with a Constant")
            if replace_missing_enabled:
                replace_missing_cols = st.multiselect("Select columns to replace missing values", options=list(df.columns))
                constant_value = st.text_input("Constant value to fill missing values", value="0")
            missing_indicator_enabled = st.checkbox("Add Missing Value Indicator for columns")
            if missing_indicator_enabled:
                missing_indicator_cols = st.multiselect("Select columns to add missing indicator", options=list(df.columns))
            robust_scaling_enabled = st.checkbox("Apply Robust Scaling to numeric columns")
            if robust_scaling_enabled:
                robust_scaling_cols = st.multiselect("Select numeric columns for robust scaling", options=num_cols)
            rank_enabled = st.checkbox("Apply Rank Transformation to numeric columns")
            if rank_enabled:
                rank_cols = st.multiselect("Select numeric columns for rank transformation", options=num_cols)
            log1p_enabled = st.checkbox("Apply Log1p Transformation")
            if log1p_enabled:
                log1p_cols = st.multiselect("Select numeric columns for log1p transformation", options=num_cols)
            poly_interaction_enabled = st.checkbox("Apply Interaction Only Polynomial Features")
            if poly_interaction_enabled:
                poly_interaction_cols = st.multiselect("Select numeric columns for interaction-only polynomial features", options=num_cols)
            custom_transform_enabled = st.checkbox("Apply Custom Transformation (enter Python code)")
            if custom_transform_enabled:
                custom_code = st.text_area("Enter custom Python code. Use 'df' as the dataframe variable.", height=150)
            tfidf_enabled = st.checkbox("Apply TF-IDF Vectorization for Text Columns")
            if tfidf_enabled:
                tfidf_cols = st.multiselect("Select text columns for TF-IDF", options=list(df.select_dtypes(include=["object"]).columns))
                max_features = st.number_input("Max features for TF-IDF", min_value=10, max_value=1000, value=100, step=10)
    
            st.markdown("### Dimensionality Reduction Options")
            dim_red_enabled = st.checkbox("Apply Dimensionality Reduction")
            if dim_red_enabled:
                dr_method = st.selectbox("Select Dimensionality Reduction Method", 
                                          ["PCA", "Kernel PCA", "Truncated SVD", "t-SNE", "UMAP" if umap_available else "t-SNE", "LDA"])
                n_components = st.number_input("Number of Components", min_value=1, max_value=50, value=2, step=1)
                dr_cols = st.multiselect("Select columns for dimensionality reduction", options=list(df.columns), 
                                         default=list(df.select_dtypes(include=["number"]).columns))
    
            if st.button("Apply Feature Engineering"):
                try:
                    df_fe = df.copy()
                    if drop_cols:
                        df_fe.drop(columns=drop_cols, inplace=True)
                    for col in log_cols:
                        try:
                            df_fe[f"{col}_log"] = df_fe[col].apply(lambda x: np.log(x) if (x is not None and x > 0) else np.nan)
                        except Exception as e:
                            st.error(f"Error in log transform {col}: {e}")
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
                                st.error(f"Error in polynomial features for {col}: {e}")
                    if binning_enabled and bin_cols:
                        for col in bin_cols:
                            try:
                                if bin_method == "Equal Width":
                                    df_fe[f"{col}_binned"] = pd.cut(df_fe[col], bins=n_bins, labels=False)
                                else:
                                    df_fe[f"{col}_binned"] = pd.qcut(df_fe[col], q=n_bins, labels=False, duplicates='drop')
                            except Exception as e:
                                st.error(f"Error in binning {col}: {e}")
                    if interaction_enabled and len(interaction_cols) > 1:
                        try:
                            df_fe["interaction_" + "_".join(interaction_cols)] = df_fe[interaction_cols].prod(axis=1)
                        except Exception as e:
                            st.error(f"Error in interaction features: {e}")
                    if sqrt_enabled and sqrt_cols:
                        for col in sqrt_cols:
                            try:
                                df_fe[f"{col}_sqrt"] = df_fe[col].apply(lambda x: np.sqrt(x) if x >= 0 else np.nan)
                            except Exception as e:
                                st.error(f"Error in sqrt transform for {col}: {e}")
                    if exp_enabled and exp_cols:
                        for col in exp_cols:
                            try:
                                df_fe[f"{col}_exp"] = np.exp(df_fe[col])
                            except Exception as e:
                                st.error(f"Error in exponential transform for {col}: {e}")
                    if power_enabled and power_cols:
                        try:
                            from sklearn.preprocessing import PowerTransformer
                            pt = PowerTransformer(method="yeo-johnson")
                            df_power = pd.DataFrame(pt.fit_transform(df_fe[power_cols]), columns=[f"{col}_power" for col in power_cols], index=df_fe.index)
                            df_fe = pd.concat([df_fe, df_power], axis=1)
                        except Exception as e:
                            st.error(f"Error in power transformation: {e}")
                    if outlier_enabled and outlier_cols:
                        try:
                            for col in outlier_cols:
                                Q1 = df_fe[col].quantile(0.25)
                                Q3 = df_fe[col].quantile(0.75)
                                IQR = Q3 - Q1
                                df_fe = df_fe[(df_fe[col] >= Q1 - iqr_multiplier * IQR) & (df_fe[col] <= Q3 + iqr_multiplier * IQR)]
                        except Exception as e:
                            st.error(f"Error in outlier removal for {col}: {e}")
                    if cross_enabled and len(cross_cols) == 2:
                        try:
                            df_fe[f"cross_{cross_cols[0]}_{cross_cols[1]}"] = df_fe[cross_cols[0]].astype(str) + "_" + df_fe[cross_cols[1]].astype(str)
                        except Exception as e:
                            st.error(f"Error in feature crossing: {e}")
                    if freq_enabled and freq_cols:
                        for col in freq_cols:
                            try:
                                freq = df_fe[col].value_counts()
                                df_fe[f"{col}_freq"] = df_fe[col].map(freq)
                            except Exception as e:
                                st.error(f"Error in frequency encoding for {col}: {e}")
                    if target_encode_enabled and target_encode_cols and "target_column" in st.session_state:
                        for col in target_encode_cols:
                            try:
                                mapping = df_fe.groupby(col)[st.session_state["target_column"]].mean()
                                df_fe[f"{col}_target_enc"] = df_fe[col].map(mapping)
                            except Exception as e:
                                st.error(f"Error in target encoding for {col}: {e}")
                    if datetime_enabled and datetime_cols:
                        for col in datetime_cols:
                            try:
                                df_fe[col] = pd.to_datetime(df_fe[col], errors='coerce')
                                df_fe[f"{col}_year"] = df_fe[col].dt.year
                                df_fe[f"{col}_month"] = df_fe[col].dt.month
                                df_fe[f"{col}_day"] = df_fe[col].dt.day
                                df_fe[f"{col}_weekday"] = df_fe[col].dt.weekday
                            except Exception as e:
                                st.error(f"Error in datetime extraction for {col}: {e}")
                    if text_len_enabled and text_cols:
                        for col in text_cols:
                            try:
                                df_fe[f"{col}_len"] = df_fe[col].astype(str).apply(len)
                            except Exception as e:
                                st.error(f"Error in text length feature for {col}: {e}")
                    if replace_missing_enabled and replace_missing_cols:
                        for col in replace_missing_cols:
                            try:
                                df_fe[col] = df_fe[col].fillna(constant_value)
                            except Exception as e:
                                st.error(f"Error in replacing missing for {col}: {e}")
                    if missing_indicator_enabled and missing_indicator_cols:
                        for col in missing_indicator_cols:
                            try:
                                df_fe[f"{col}_missing"] = df_fe[col].isna().astype(int)
                            except Exception as e:
                                st.error(f"Error in missing indicator for {col}: {e}")
                    if robust_scaling_enabled and robust_scaling_cols:
                        try:
                            from sklearn.preprocessing import RobustScaler
                            rs = RobustScaler()
                            df_rs = pd.DataFrame(rs.fit_transform(df_fe[robust_scaling_cols]), columns=[f"{col}_robust" for col in robust_scaling_cols], index=df_fe.index)
                            df_fe = pd.concat([df_fe, df_rs], axis=1)
                        except Exception as e:
                            st.error(f"Error in robust scaling: {e}")
                    if rank_enabled and rank_cols:
                        for col in rank_cols:
                            try:
                                df_fe[f"{col}_rank"] = df_fe[col].rank()
                            except Exception as e:
                                st.error(f"Error in rank transformation for {col}: {e}")
                    if log1p_enabled and log1p_cols:
                        for col in log1p_cols:
                            try:
                                df_fe[f"{col}_log1p"] = np.log1p(df_fe[col])
                            except Exception as e:
                                st.error(f"Error in log1p transformation for {col}: {e}")
                    if poly_interaction_enabled and poly_interaction_cols:
                        try:
                            from sklearn.preprocessing import PolynomialFeatures
                            poly_int = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                            poly_int_features = poly_int.fit_transform(df_fe[poly_interaction_cols])
                            poly_int_feature_names = poly_int.get_feature_names_out(poly_interaction_cols)
                            poly_int_df = pd.DataFrame(poly_int_features, columns=[f"{name}_int" for name in poly_int_feature_names], index=df_fe.index)
                            df_fe = pd.concat([df_fe, poly_int_df], axis=1)
                        except Exception as e:
                            st.error(f"Error in interaction-only polynomial features: {e}")
                    if custom_transform_enabled and custom_code.strip() != "":
                        try:
                            exec(custom_code, {'df': df_fe, 'np': np, 'pd': pd})
                        except Exception as e:
                            st.error(f"Error in custom transformation: {e}")
                    if tfidf_enabled and tfidf_cols:
                        try:
                            for col in tfidf_cols:
                                vectorizer = TfidfVectorizer(max_features=int(max_features))
                                tfidf_matrix = vectorizer.fit_transform(df_fe[col].astype(str))
                                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(tfidf_matrix.shape[1])], index=df_fe.index)
                                df_fe = pd.concat([df_fe.drop(columns=[col]), tfidf_df], axis=1)
                        except Exception as e:
                            st.error(f"Error in TF-IDF vectorization for {col}: {e}")
                    if dim_red_enabled and dr_cols:
                        try:
                            if dr_method == "PCA":
                                dr_model = PCA(n_components=int(n_components))
                            elif dr_method == "Kernel PCA":
                                dr_model = KernelPCA(n_components=int(n_components))
                            elif dr_method == "Truncated SVD":
                                dr_model = TruncatedSVD(n_components=int(n_components))
                            elif dr_method == "t-SNE":
                                dr_model = TSNE(n_components=int(n_components))
                            elif dr_method == "UMAP" and umap_available:
                                dr_model = umap.UMAP(n_components=int(n_components))
                            elif dr_method == "LDA":
                                
                                dr_model = LinearDiscriminantAnalysis(n_components=int(n_components))
                            else:
                                dr_model = PCA(n_components=int(n_components))
                            dr_features = dr_model.fit_transform(df_fe[dr_cols])
                            dr_df = pd.DataFrame(dr_features, columns=[f"{dr_method}_comp_{i}" for i in range(1, int(n_components)+1)], index=df_fe.index)
                            df_fe = df_fe.drop(columns=dr_cols)
                            df_fe = pd.concat([df_fe, dr_df], axis=1)
                            if int(n_components) >= 2:
                                fig, ax = plt.subplots()
                                ax.scatter(dr_df.iloc[:,0], dr_df.iloc[:,1])
                                ax.set_xlabel("Component 1")
                                ax.set_ylabel("Component 2")
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error in dimensionality reduction: {e}")
                    st.session_state.df = sanitize_column_names(df_fe)
                    st.session_state.feature_engineered = True
                    st.success("Features applied!")
                except Exception as e:
                    st.error(f"Feature engineering failed: {str(e)}")

                            
                df_fe = sanitize_column_names(df_fe)    
                st.session_state["df"] = df_fe
                st.success("Feature engineering applied successfully!")
                st.dataframe(df_fe.head())
    
# =====================================================
# Tab 3: Model Training & Tuning
# =====================================================
    with tabs[2]:
        st.header("Model Training & Hyperparameter Tuning")
        if st.session_state.df is None or st.session_state.df.empty:
            st.warning("No dataset found or dataset is empty. Please upload data in the Data Upload tab.")
        else:
            df = st.session_state["df"]
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            problem_type = st.radio("Select Problem Type", options=["Classification", "Regression"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                target_column = st.selectbox("Select the Target Column", list(df.columns))
                st.session_state["target_column"] = target_column
            with col2:
                scaler_options = ["standard", "minmax", "none"]
                scaler_type = st.selectbox("Select Scaler Type", scaler_options)
            with col3:
                imputer_method = st.selectbox("Select Imputation Method", ["mean", "knn", "iterative"] if problem_type=="Classification" else ["median", "knn", "iterative"])

            st.markdown("### Model Options")
            ensemble_enabled = st.checkbox("Use Ensemble Methods (Stacking)")

            # Add the unused models to the classification model_dict
            if problem_type == "Classification":
                model_dict = {
                    "Logistic Regression": {
                        "model": LogisticRegression(),
                        "default_grid": {"C": [0.1, 1, 10], "solver": ["lbfgs", "liblinear"], "max_iter": [100, 200, 500]}
                    },
                    "Random Forest Classifier": {
                        "model": RandomForestClassifier(),
                        "default_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}
                    },
                    "SVC": {
                        "model": SVC(probability=True),
                        "default_grid": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear", "poly"]}
                    },
                    "Extra Trees Classifier": {
                        "model": ExtraTreesClassifier(),
                        "default_grid": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20, 30]}
                    },
                    "AdaBoost Classifier": {
                        "model": AdaBoostClassifier(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
                    },
                    "Linear SVC": {
                        "model": LinearSVC(),
                        "default_grid": {"C": [0.1, 1, 10], "max_iter": [1000, 2000]}
                    },
                    "NuSVC": {
                        "model": NuSVC(probability=True),
                        "default_grid": {"nu": [0.1, 0.5, 0.7], "kernel": ["rbf", "linear", "poly"]}
                    },
                    "Quadratic Discriminant Analysis": {
                        "model": QuadraticDiscriminantAnalysis(),
                        "default_grid": {}
                    },
                    "Bernoulli Naive Bayes": {
                        "model": BernoulliNB(),
                        "default_grid": {"alpha": [0.1, 0.5, 1.0]}
                    },
                }
                if xgb_available:
                    model_dict["XGBoost Classifier"] = {
                        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
                    }
                model_dict.update({
                    "Decision Tree Classifier": {
                        "model": DecisionTreeClassifier(),
                        "default_grid": {"max_depth": [None, 5, 10, 20]}
                    },
                    "K-Nearest Neighbors": {
                        "model": KNeighborsClassifier(),
                        "default_grid": {"n_neighbors": [3, 5, 7, 9]}
                    },
                    "Gradient Boosting Classifier": {
                        "model": GradientBoostingClassifier(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
                    },
                    "Linear Discriminant Analysis": {
                        "model": LinearDiscriminantAnalysis(),
                        "default_grid": {}
                    },
                    "GaussianNB": {
                        "model": GaussianNB(),
                        "default_grid": {}
                    }
                })
                if ensemble_enabled:
                    estimators = [(name, m["model"]) for name, m in model_dict.items()]
                    ensemble_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
                    model_dict = {"Stacking Classifier": {"model": ensemble_model, "default_grid": {}}}
                    
            else:
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
                        "default_grid": {"n_neighbors": [3, 5, 7, 9]}
                    },
                    "Gradient Boosting Regressor": {
                        "model": GradientBoostingRegressor(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 5, 7]}
                    },
                }
                if xgb_available:
                    model_dict["XGBoost Regressor"] = {
                        "model": XGBRegressor(),
                        "default_grid": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]}
                    }
                if ensemble_enabled:
                    estimators = [(name, m["model"]) for name, m in model_dict.items()]
                    ensemble_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
                    model_dict = {"Stacking Regressor": {"model": ensemble_model, "default_grid": {}}}

            model_name = st.text_input("Enter a name for the model", value="my_model")
            model_version = st.text_input("Enter model version", value="1.0")
            selected_model_name = st.selectbox("Select a Model", list(model_dict.keys()))
            base_model = model_dict[selected_model_name]["model"]
            default_grid = model_dict[selected_model_name]["default_grid"]

            st.markdown("### Hyperparameter Tuning Options")
            tuning_enabled = st.checkbox("Perform Hyperparameter Tuning", value=False)
            if tuning_enabled:
                search_method_choice = st.radio("Select Tuning Method", options=["GridSearchCV", "RandomizedSearchCV", "HalvingGridSearchCV", "HalvingRandomSearchCV", "Optuna"], index=0)
                cv_folds = st.number_input("Number of CV folds", min_value=2, max_value=10, value=5, step=1)
                n_iter = 10
                if search_method_choice in ["RandomizedSearchCV", "HalvingRandomSearchCV", "Optuna"]:
                    n_iter = st.number_input("Number of parameter settings sampled", min_value=1, max_value=100, value=10, step=1)
                param_grid_str = st.text_area("Parameter Grid (JSON format)-> you can modify this file", value=json.dumps(default_grid, indent=4))
                try:
                    param_grid = json.loads(param_grid_str) if param_grid_str.strip() != "" else default_grid
                except Exception as e:
                    st.error(f"Error parsing parameter grid: {e}")
                    param_grid = default_grid
            else:
                pass  # Manual parameter adjustments can be added here.

            if st.button("Train Model"):
                df = st.session_state["df"]
                df = sanitize_column_names(df)
                st.session_state["df"] = df
                
                progress_bar = st.progress(0)
                if problem_type == "Classification":
                    x_train, x_test, y_train, y_test = preprocess_classification_data(df, target_column, scaler_type, imputer_method)
                else:
                    x_train, x_test, y_train, y_test = preprocess_regression_data(df, target_column, scaler_type, imputer_method)
                st.session_state.x_test = x_test 
                progress_bar.progress(30)

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
                        elif search_method_choice == "Optuna":
                            method = "optuna"
                        best_model, best_params = tune_model(x_train, y_train, base_model, param_grid, search_method=method, cv=cv_folds, n_iter=int(n_iter))
                    st.success(f"Best Parameters: {best_params}")
                    trained_model = best_model
                else:
                    trained_model = base_model
                    trained_model.fit(x_train, y_train)
                    
                st.session_state.trained_model = trained_model 
                progress_bar.progress(70)
                
                progress_bar.progress(90)
                metrics = evaluation(trained_model, x_test, y_test, problem_type)

                # Display metrics based on problem type
                if problem_type == "Classification":
                    st.success(f"Model '{model_name}' trained with the following metrics:")
                    st.write(f"- Accuracy: {metrics['accuracy']}")
                    st.write(f"- Precision: {metrics['precision']}")
                    st.write(f"- Recall: {metrics['recall']}")
                    st.write(f"- F1 Score: {metrics['f1_score']}")
                    st.write("Confusion Matrix:")
                    st.write(metrics["confusion_matrix"])
                else:
                    st.success(f"Model '{model_name}' trained with the following metrics:")
                    st.write(f"- R²: {metrics['r2']}")
                    st.write(f"- MAE: {metrics['mae']}")
                    st.write(f"- RMSE: {metrics['rmse']}")
                    st.write(f"- MSE: {metrics['mse']}")
                progress_bar.progress(100)
                
        col1, col2 = st.columns(2)
        with col1:
            try:
                model_bytes = pickle.dumps(st.session_state.trained_model)
                st.download_button(
                    "Download Trained Model",
                    data=model_bytes,
                    file_name=f"{model_name}_v{model_version}.pkl",
                    mime="application/octet-stream"
                )
            except Exception as e:
                st.error(f"Download failed: {str(e)}")

        with col2:
            if st.button("Upload Model to Cloud"):
                if "trained_model" not in st.session_state or st.session_state.trained_model is None:
                    st.error("No trained model found! Train a model first.")
                else:
                    st.info("The cloud functionality is not implemented yet")
                            # Explicit state preservation
                    st.session_state.update({
                        "df": st.session_state.df,
                        "trained_model": st.session_state.trained_model,
                        "x_test": st.session_state.x_test
                    })

                # SHAP Explanation (separate section)
        st.markdown("---")                
        # SHAP Explanation
        

        if st.button("Explain Model"):
            if "trained_model" not in st.session_state or st.session_state.trained_model is None:
                st.error("Train a model first!")
        else:
            with st.spinner("Generating explanation..."):
                try:
                # Get sample data
                    sample_data = st.session_state.x_test.iloc[:50]  # Use first 50 samples
                
                # Initialize appropriate explainer based on model type
                    model_type = type(st.session_state.trained_model).__name__
                
                # SHAP explainer selection
                    if model_type in ["RandomForestClassifier", "RandomForestRegressor",
                                 "XGBClassifier", "XGBRegressor", 
                                 "GradientBoostingClassifier", "GradientBoostingRegressor"]:
                        explainer = shap.TreeExplainer(st.session_state.trained_model)
                        shap_values = explainer.shap_values(sample_data)
                    elif model_type in ["LogisticRegression", "LinearRegression"]:
                        explainer = shap.LinearExplainer(st.session_state.trained_model, sample_data)
                        shap_values = explainer.shap_values(sample_data)
                    else:
                    # Use KernelExplainer as fallback
                        explainer = shap.KernelExplainer(st.session_state.trained_model.predict, sample_data)
                        shap_values = explainer.shap_values(sample_data)
                
                # Visualization
                    st.subheader(f"Model Explanation ({model_type})")
                
                # Summary plot
                    fig1, ax1 = plt.subplots()
                    shap.summary_plot(shap_values, sample_data, show=False)
                    st.pyplot(fig1)
                    plt.close(fig1)
                
                # Feature importance
                    fig2, ax2 = plt.subplots()
                    shap.summary_plot(shap_values, sample_data, plot_type="bar", show=False)
                    st.pyplot(fig2)
                    plt.close(fig2)
                
                # Individual prediction explanations
                    st.markdown("### Individual Prediction Explanations")
                    sample_idx = st.selectbox("Select sample to explain", range(len(sample_data)))
                    fig3, ax3 = plt.subplots()
                    shap.force_plot(explainer.expected_value, 
                              shap_values[sample_idx,:], 
                              sample_data.iloc[sample_idx,:], 
                              show=False, matplotlib=True)
                    st.pyplot(fig3)
                    plt.close(fig3)
                
                # Decision plot
                    st.markdown("### Decision Process Visualization")
                    fig4, ax4 = plt.subplots()
                    shap.decision_plot(explainer.expected_value, 
                                 shap_values[sample_idx,:], 
                                 sample_data.iloc[sample_idx,:], 
                                 show=False)
                    st.pyplot(fig4)
                    plt.close(fig4)
                
                except Exception as e:
                    st.error(f"Explanation failed: {str(e)}")
                    st.markdown("""
                **Common Solutions:**
                - For non-tree models, try with smaller datasets
                - Some models (like KNN) may not support SHAP explanations
                - Consider permutation importance as alternative:
                """)
                
                # Fallback to permutation importance
                    from sklearn.inspection import permutation_importance
                    with st.spinner("Calculating permutation importance..."):
                        try:
                            result = permutation_importance(st.session_state.trained_model,
                                                      st.session_state.x_test,
                                                      st.session_state.y_test,
                                                      n_repeats=10,
                                                      random_state=42)
                            importance_df = pd.DataFrame({
                                'features': st.session_state.x_test.columns,
                                'importance_mean': result.importances_mean,
                                'importance_std': result.importances_std
                            }).sort_values('importance_mean', ascending=False)
                        
                            st.subheader("Permutation Importance")
                            st.dataframe(importance_df)
                        
                            fig5, ax5 = plt.subplots()
                            importance_df.plot.bar(x='features', y='importance_mean', 
                                             yerr='importance_std', ax=ax5)
                            st.pyplot(fig5)
                        
                        except Exception as perm_error:
                            st.error(f"Fallback explanation also failed: {str(perm_error)}")
                                
if __name__ == "__main__":
    run()
