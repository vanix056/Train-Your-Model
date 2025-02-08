# util.py
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


def read_data(file):
    """Reads data from a file (CSV or Excel)."""
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        try:
            df = pd.read_excel(file)
            return df
        except Exception as e:
            print(f"Error reading file: {e}")
            return None


def preprocess_data(df, target_col, scaler_type):
    """Preprocesses the data: handles missing values, splits data, and scales numerical features."""
    x = df.drop(columns=target_col)
    y = df[target_col]

    num_cols = x.select_dtypes(include=['number']).columns
    cat_cols = x.select_dtypes(include=['object', 'category']).columns

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Numerical column processing
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy='mean')
        x_train[num_cols] = num_imputer.fit_transform(x_train[num_cols])
        x_test[num_cols] = num_imputer.transform(x_test[num_cols])

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        # One-Hot Encoding for numerical features is not typical; skipping
        else:
            scaler = None  # No scaling

        if scaler:  # Apply scaler only if it's defined
            x_train[num_cols] = scaler.fit_transform(x_train[num_cols])
            x_test[num_cols] = scaler.transform(x_test[num_cols])

    # Categorical column processing
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        x_train[cat_cols] = cat_imputer.fit_transform(x_train[cat_cols])
        x_test[cat_cols] = cat_imputer.transform(x_test[cat_cols])

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        encoder.fit(x_train[cat_cols])
        x_train_enc = encoder.transform(x_train[cat_cols])
        x_test_enc = encoder.transform(x_test[cat_cols])

        x_train_enc = pd.DataFrame(x_train_enc, index=x_train.index,
                                     columns=encoder.get_feature_names_out(cat_cols))
        x_test_enc = pd.DataFrame(x_test_enc, index=x_test.index,
                                    columns=encoder.get_feature_names_out(cat_cols))

        x_train = pd.concat([x_train.drop(columns=cat_cols), x_train_enc], axis=1)
        x_test = pd.concat([x_test.drop(columns=cat_cols), x_test_enc], axis=1)

    return x_train, x_test, y_train, y_test


def model_train(x_train, y_train, model, model_name):
    """Trains the given model and saves it to a file."""
    model.fit(x_train, y_train)
    model_path = os.path.join(parent_dir, 'trained_model', f'{model_name}.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    return model


def evaluation(model, x_test, y_test):
    """Evaluates the model and returns the accuracy."""
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    acc = round(acc, 2)
    return acc
