# app.py
import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform, uniform
import pandas as pd
import base64  # Import base64
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from util import read_data, preprocess_data, model_train, evaluation

# Working directory setup
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

# Streamlit app configuration
st.set_page_config(page_title='No Code ML Training', page_icon="🧠", layout="centered")

# Enhanced GUI with Streamlit
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Home", "Training", "About"],
        icons=['house', 'gear', 'info-circle'],
        menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#262730"},
            "icon": {"color": "white", "font-size": "25px"},
            "nav-link": {"font-size": "20px", "text-align": "left", "margin": "0px", "--hover-color": "#4F8BF9"},
            "nav-link-selected": {"background-color": "#4F8BF9"},
        }
    )

if selected == "Home":
    st.title("No Code ML Training")
    st.markdown("Welcome to the No Code ML Training App! Upload your data, select your model, and train with ease.")

elif selected == "Training":
    st.title("ML Model Training")

    # Option to upload file
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Read data based on file type
        try:
            df = read_data(uploaded_file)
            st.dataframe(df.head())

            # Columns for selection
            col1, col2, col3, col4 = st.columns(4)

            # Scaler options
            scaler_type_list = ['standard', 'minmax', 'one hot encoder']

            # Model dictionary with more models
            model_dic = {
                'Logistic Regression': LogisticRegression(),
                'Random Forest Classifier': RandomForestClassifier(),
                'Support Vector Classifier': SVC(),
                'XGBoost Classifier': XGBClassifier(),
                'Decision Tree Classifier': DecisionTreeClassifier(),  # added
                'Gradient Boosting Classifier': GradientBoostingClassifier()  # added
            }

            with col1:
                target_coloumn = st.selectbox("Select the Target Column", list(df.columns))
            with col2:
                scaler_type = st.selectbox("Select Scaler Type", scaler_type_list)
            with col3:
                selected_model = st.selectbox("Select a Model", list(model_dic.keys()))
            with col4:
                model_name = st.text_input("Model name")

            # Hyperparameter tuning options
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning")

            if hyperparameter_tuning:
                tuning_method = st.selectbox("Tuning Method", ["Grid Search", "Random Search"])
                n_iter = st.number_input("Number of iterations", min_value=1, value=10)

            # Train Model button
            if st.button('Train Model'):
                # Preprocess data
                x_train, x_test, y_train, y_test = preprocess_data(df, target_coloumn, scaler_type)

                # Select model
                model = model_dic[selected_model]

                # Hyperparameter tuning
                if hyperparameter_tuning:
                    param_grid = {}
                    if selected_model == 'Logistic Regression':
                        param_grid = {'C': uniform(0.1, 10), 'penalty': ['l1', 'l2', 'elasticnet', None], 'solver': ['liblinear', 'saga']}
                    elif selected_model == 'Random Forest Classifier':
                        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5, 10]}
                    elif selected_model == 'Support Vector Classifier':
                        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                    elif selected_model == 'XGBoost Classifier':
                        param_grid = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]}

                    if tuning_method == "Grid Search":
                        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
                        grid_search.fit(x_train, y_train)
                        best_model = grid_search.best_estimator_
                    elif tuning_method == "Random Search":
                        random_search = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=3, scoring='accuracy')
                        random_search.fit(x_train, y_train)
                        best_model = random_search.best_estimator_
                    model = best_model  # Use the best model from tuning
                # Train the model
                train = model_train(x_train, y_train, model, model_name)

                # Evaluate the model
                acc = evaluation(model, x_test, y_test)

                st.success(f"Model {model_name} trained successfully with a Test Accuracy of " + str(acc))

                # Download model
                model_path = os.path.join(parent_dir, 'trained_model', f'{model_name}.pkl')
                with open(model_path, 'rb') as f:
                    model_bytes = f.read()
                b64 = base64.b64encode(model_bytes).decode()
                href = f'<a href="data:file/pkl;base64,{b64}" download="{model_name}.pkl">Download Trained Model</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif selected == "About":
    st.title("About")
    st.markdown("This is a no-code machine learning training app built with Streamlit.  It allows you to train various ML models without writing code.")
