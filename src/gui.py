import os
import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie

# Import models and utility functions
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from util import preprocess_data, model_train, evaluation

# --- Helper function for Lottie Animations ---
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# --- Main GUI Function ---
def run():
    # Set page configuration
    st.set_page_config(page_title="No Code ML Training", page_icon="🧠", layout="wide")
    
    # Display a Lottie animation in the header
    lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_tll0j4bb.json")
    if lottie_animation:
        st_lottie(lottie_animation, height=200)
    
    st.title("No Code ML Training")
    
    # --- File Upload Section ---
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format!")
                return
        except Exception as e:
            st.error(f"Error reading the file: {e}")
            return
    else:
        st.info("Please upload a dataset file to begin.")
        st.stop()  # stop execution until file is uploaded

    st.dataframe(df.head())

    # --- Data and Model Options ---
    col1, col2, col3 = st.columns(3)
    with col1:
        target_column = st.selectbox("Select the Target Column", list(df.columns))
    with col2:
        # Options for numeric scaling (we avoid one hot encoding here because that’s for categorical features)
        scaler_options = ['standard', 'minmax', 'none']
        scaler_type = st.selectbox("Select Scaler Type for Numeric Columns", scaler_options)
    with col3:
        model_name = st.text_input("Enter a Model Name", value="my_model")

    st.subheader("Select Model")
    # Extended model list
    model_options = [
        "Logistic Regression",
        "Random Forest Classifier",
        "Support Vector Classifier",
        "XGBoost Classifier",
        "Decision Tree Classifier",
        "K-Nearest Neighbors",
        "Gradient Boosting Classifier"
    ]
    selected_model = st.selectbox("Choose a model", model_options)

    # --- Hyperparameter Tuning in the Sidebar ---
    st.sidebar.title("Hyperparameter Tuning")
    model = None  # will be set based on selection and hyperparameter inputs

    if selected_model == "Logistic Regression":
        st.sidebar.header("Logistic Regression Parameters")
        C = st.sidebar.number_input("C (Inverse Regularization Strength)", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")
        max_iter = st.sidebar.number_input("Maximum Iterations", min_value=100, max_value=10000, value=1000, step=100)
        solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        model = LogisticRegression(C=C, max_iter=int(max_iter), solver=solver)
        
    elif selected_model == "Random Forest Classifier":
        st.sidebar.header("Random Forest Parameters")
        n_estimators = st.sidebar.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
        max_depth = st.sidebar.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = None if max_depth == 0 else int(max_depth)
        model = RandomForestClassifier(n_estimators=int(n_estimators), max_depth=max_depth, criterion=criterion)
        
    elif selected_model == "Support Vector Classifier":
        st.sidebar.header("SVC Parameters")
        C = st.sidebar.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01, format="%.2f")
        kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel, probability=True)
        
    elif selected_model == "XGBoost Classifier":
        st.sidebar.header("XGBoost Parameters")
        n_estimators = st.sidebar.number_input("Number of Trees", min_value=10, max_value=1000, value=100, step=10)
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
        model = XGBClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth),
                              use_label_encoder=False, eval_metric='logloss')
        
    elif selected_model == "Decision Tree Classifier":
        st.sidebar.header("Decision Tree Parameters")
        max_depth = st.sidebar.number_input("Max Depth (0 for None)", min_value=0, max_value=100, value=0, step=1)
        criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        max_depth = None if max_depth == 0 else int(max_depth)
        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion)
        
    elif selected_model == "K-Nearest Neighbors":
        st.sidebar.header("KNN Parameters")
        n_neighbors = st.sidebar.number_input("Number of Neighbors", min_value=1, max_value=50, value=5, step=1)
        weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors), weights=weights)
        
    elif selected_model == "Gradient Boosting Classifier":
        st.sidebar.header("Gradient Boosting Parameters")
        n_estimators = st.sidebar.number_input("Number of Estimators", min_value=10, max_value=1000, value=100, step=10)
        learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01, format="%.2f")
        max_depth = st.sidebar.number_input("Max Depth", min_value=1, max_value=20, value=3, step=1)
        model = GradientBoostingClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth))

    # --- Train Model Button ---
    if st.button("Train Model"):
        # Preprocess the data (numeric scaling and one-hot encoding for categoricals)
        x_train, x_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)
        
        # Train the selected model and save it
        trained_model = model_train(x_train, y_train, model, model_name)
        acc = evaluation(trained_model, x_test, y_test)
        st.success(f"Model '{model_name}' trained successfully with a test accuracy of {acc}")

        # --- Download Trained Model ---
        # Build the model path (assuming util.py saves the model in ../trained_model/)
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(parent_dir, "trained_model", f"{model_name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Trained Model",
                    data=f.read(),
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"
                )
        else:
            st.error("Model file not found. Something went wrong.")

if __name__ == "__main__":
    run()
