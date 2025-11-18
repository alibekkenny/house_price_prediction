from sklearn.feature_selection import SelectKBest, f_regression, RFE

def add_features(df):
    df = df.copy()

    # возраст дома
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # есть ли гараж
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)

    return df

def select_k_best_features(X_train, y_train, X_val, k=10):
    selector = SelectKBest(score_func=f_regression, k=k)
    
    X_train_kbest = selector.fit_transform(X_train, y_train)
    X_val_kbest = selector.transform(X_val)

    return X_train_kbest, X_val_kbest, selector

def select_rfe_features(model, X_train, y_train, X_val, n_features=10):
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_val_rfe = rfe.transform(X_val)

    return X_train_rfe, X_val_rfe, rfe
