import os
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from util import read_data,preprocess_data,model_train,evaluation

working_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(working_dir)


st.set_page_config(title="No Code ML",icon="🧠",layout="centered")
st.title("No Code ML Training")
dataset_list=os.listdir(f'{parent_dir}/data')

dataset=st.selectbox("Select the dataset to train",dataset_list,index=None)


df = read_data(dataset)



