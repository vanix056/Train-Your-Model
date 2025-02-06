import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

working_dir=os.path.dirname(os.path.abspath(__file__))
parent_dir=os.path.dirname(working_dir)


def read_data(dataset):
    data_path = f'{parent_dir}/data/{dataset}'
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        return df
    elif data_path.endswith('.xlsx') or data_path.endswith('xls'):
        df = pd.read_excel(data_path)
        return df
        
    
def preprocess_data(df,target_col,scaler_type_list):
    x=df.drop(columns=target_col)
    y=df[target_col]
    
    num_cols=x.select_dtypes(include=['number']).columns
    cat_cols=x.select_dtypes(include=['object','category']).columns
    
    if len(num_cols)==0:
        pass
    else:
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
        num_imputer=SimpleImputer(strategy='mean')
        x_train[num_cols]=num_imputer.fit_transform(x_train[num_cols])
        x_test[num_cols]=num_imputer.transform(x_test[num_cols])
        
        if scaler_type_list=='standard':
            scaler=StandardScaler()
        elif scaler_type_list=='minmax':
            scaler=MinMaxScaler()
        elif scaler_type_list=='one hot encoder':
            scaler=OneHotEncoder()
        
        x_train[num_cols]=scaler.fit_transform(x_train[num_cols])
        x_test[num_cols]=scaler.transform(x_test[num_cols])
    
    if len(cat_cols)==0:
        pass
    else:
        cat_imputer=SimpleImputer(strategy='most frequent')
        x_train[cat_cols]=cat_imputer.fit_transform(x_train[cat_cols])
        x_test[cat_cols]=cat_imputer.transform(x_test[cat_cols])
        
        encoder=OneHotEncoder()
        x_train_enc=encoder.fit_transform(x_train[cat_cols])
        x_test_enc=encoder.transform(x_test[cat_cols])
        x_train_enc=pd.DataFrame(x_train_enc.toarray(),columns=encoder.get_feature_names_(cat_cols))
        x_test_enc=pd.DataFrame(x_test_enc.toarray(),columns=encoder.get_feature_names(cat_cols))
        x_train=pd.concat([x_train.drop(columns=cat_cols),x_train_enc],axis=1)
        x_test=pd.concat([x_test.drop(columns=cat_cols),x_test_enc],axis=1)
        
    return x_train,x_test,y_train,y_test
    
        

def model_train(x_train,y_train,model,model_name):
    model.fit(x_train,y_train)
    with open(f'{parent_dir}/trained_model/{model_name}.pkl','wb') as f:
        pickle.dump(model,f)
    return model
    
    
def evaluation(model,x_test,y_test):
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    acc=round(acc,2)
    
    return acc
