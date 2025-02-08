import streamlit as st
from streamlit.components.v1 import html

def set_custom_style():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        * {
            font-family: 'Roboto', sans-serif;
        }
        
        .title-text {
            font-size: 2.8rem !important;
            color: #2c3e50;
            text-align: center;
            animation: slideIn 1s ease-in-out;
            margin-bottom: 30px;
        }
        
        @keyframes slideIn {
            0% { transform: translateY(-50px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 25px;
            font-size: 1rem;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        }
        
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 12px !important;
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(45deg, #4CAF50, #45a049);
        }
    </style>
    """, unsafe_allow_html=True)

def file_upload_section():
    return st.file_uploader(
        "📤 Upload Dataset (CSV/Excel)",
        type=["csv", "xlsx", "xls"],
        help="Drag and drop your dataset file here or click to browse"
    )

def model_selection_ui(models_config):
    selected_model = st.selectbox(
        "🤖 Select Model",
        list(models_config.keys()),
        help="Choose a machine learning algorithm"
    )
    
    params = {}
    with st.expander("⚙️ Hyperparameter Tuning", expanded=False):
        for param, config in models_config[selected_model]['params'].items():
            if config['type'] == 'slider':
                params[param] = st.slider(
                    label=config['label'],
                    min_value=config['min'],
                    max_value=config['max'],
                    value=config['default'],
                    step=config['step'],
                    help=config['help']
                )
            elif config['type'] == 'select':
                params[param] = st.selectbox(
                    label=config['label'],
                    options=config['options'],
                    index=config['default_index'],
                    help=config['help']
                )
    return selected_model, params