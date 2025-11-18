from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd 

def clean_data(df):
    df = df.fillna(df.median(numeric_only=True))
    return df

def split_features_target(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return X, y

def train_val_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, scaler

def encoding_categorical(df, categorical_cols):
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)