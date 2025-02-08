import os
import time
import streamlit as st
import pandas as pd
from util import *
from gui import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

MODEL_CONFIG = {
    'Logistic Regression': {
        'class': LogisticRegression,
        'params': {
            'C': {
                'type': 'slider',
                'label': 'Regularization Strength (C)',
                'min': 0.01,
                'max': 10.0,
                'default': 1.0,
                'step': 0.1,
                'help': 'Inverse of regularization strength'
            },
            'penalty': {
                'type': 'select',
                'label': 'Penalty Type',
                'options': ['l2', 'none'],
                'default_index': 0,
                'help': 'Regularization penalty type'
            }
        }
    },
    'Support Vector Classifier': {
        'class': SVC,
        'params': {
            'C': {
                'type': 'slider',
                'label': 'Regularization (C)',
                'min': 0.1,
                'max': 10.0,
                'default': 1.0,
                'step': 0.1,
                'help': 'Regularization parameter'
            },
            'kernel': {
                'type': 'select',
                'label': 'Kernel Type',
                'options': ['linear', 'rbf', 'poly'],
                'default_index': 1,
                'help': 'Kernel type for SVM'
            }
        }
    },
    'Random Forest Classifier': {
        'class': RandomForestClassifier,
        'params': {
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Trees',
                'min': 10,
                'max': 200,
                'default': 100,
                'step': 10,
                'help': 'Number of trees in the forest'
            },
            'max_depth': {
                'type': 'slider',
                'label': 'Max Depth',
                'min': 1,
                'max': 20,
                'default': 10,
                'step': 1,
                'help': 'Maximum depth of the trees'
            }
        }
    },
    'XGBoost Classifier': {
        'class': XGBClassifier,
        'params': {
            'learning_rate': {
                'type': 'slider',
                'label': 'Learning Rate',
                'min': 0.01,
                'max': 1.0,
                'default': 0.3,
                'step': 0.01,
                'help': 'Boosting learning rate'
            },
            'n_estimators': {
                'type': 'slider',
                'label': 'Number of Trees',
                'min': 50,
                'max': 1000,
                'default': 100,
                'step': 50,
                'help': 'Number of gradient boosted trees'
            }
        }
    }
}

def main():
    # GUI Setup
    set_custom_style()
    st.markdown('<p class="title-text">AutoML Training Platform</p>', unsafe_allow_html=True)
    
    # File Upload Section
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            dataset_source = st.radio("Data Source", ["Sample Dataset", "Upload Your Own"])
        
        df = None
        if dataset_source == "Upload Your Own":
            uploaded_file = file_upload_section()
            if uploaded_file:
                df = read_uploaded_file(uploaded_file)
        else:
            with col2:
                dataset_files = os.listdir(f'{parent_dir}/data')
                dataset = st.selectbox("Select Sample Dataset", dataset_files)
                if dataset:
                    df = read_data(dataset)

    if df is not None:
        st.success("✅ Dataset loaded successfully!")
        with st.expander("📊 Data Preview", expanded=True):
            st.dataframe(df.head().style.highlight_max(axis=0, color='#d4f7d4'))

        # Data Configuration
        st.subheader("⚙️ Data Configuration")
        target_col = st.selectbox("🎯 Target Column", df.columns)
        scaler_type = st.selectbox("📏 Feature Scaling", 
                                 ['None', 'Standard Scaler', 'MinMax Scaler'])

        # Model Configuration
        st.subheader("🧠 Model Configuration")
        selected_model, params = model_selection_ui(MODEL_CONFIG)
        model_name = st.text_input("📛 Model Name", "my_model")

        # Training Section
        if st.button("🚀 Start Training"):
            with st.spinner("🔧 Training in progress..."):
                start_time = time.time()
                
                # Get model class and create instance
                model_class = MODEL_CONFIG[selected_model]['class']
                model = model_class(**params)
                
                # Preprocess data
                X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                    df, target_col, scaler_type
                )
                
                # Create and train pipeline
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('classifier', model)
                ])
                
                pipeline.fit(X_train, y_train)
                
                # Save model and preprocessing pipeline
                save_model(pipeline, model_name)
                
                # Evaluate model
                acc = evaluate_model(pipeline, X_test, y_test)
                
                # Show results
                training_time = time.time() - start_time
                st.success(f"🎉 Training completed in {training_time:.1f} seconds!")
                st.metric(label="Test Accuracy", value=f"{acc*100:.2f}%")
                
                # Download button
                with open(f'{parent_dir}/trained_models/{model_name}.pkl', "rb") as f:
                    st.download_button(
                        label="📥 Download Model",
                        data=f,
                        file_name=f"{model_name}.pkl",
                        mime="application/octet-stream"
                    )

if __name__ == "__main__":
    main()