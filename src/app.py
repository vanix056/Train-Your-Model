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

if df is not None:
    st.dataframe(df.head())
    col1,col2,col3,col4=st.coloumns(4)
    scaler_type_list=['standard','minmax','one hot encoder']
    
    model_dic={
        'Logistic Regression':LogisticRegression(),
        'Random Forest Classifier':RandomForestClassifier(),
        'Support Vector Classifier':SVC(),
        'XGBoost Classifier':XGBClassifier()
    }
    
    with col1:
        target_coloumn=st.selectbox("Select the Target Coloum",list(df.coloumns))
    with col2:
        scaler_type=st.selectbox("Select Scaler Type",scaler_type_list)
    with col3:
        selected_model=st.selectbox("Select a Model",list(model_dic.keys()))
    with col4:
        model_name=st.text_input("Model name")
        
    
    if st.button('Train Model'):
        x_train,x_test,y_train,y_test=preprocess_data(df,target_coloumn,scaler_type)
        model=model_dic[selected_model]
        train=model_train(x_train,y_train,model,model_name)
        acc=evaluation(model,x_test,y_test)
        
        st.success(f"Model {model_name} trained successfully with a Test Accuracy of  "+ str(acc))
        





