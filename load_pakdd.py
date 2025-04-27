import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_pkdd(filepath, mode='onehot'):
    """Preprocess and load PKDD dataset with support for onehot or label_encoding-based encoding."""

    df = pd.read_csv(filepath)
    df = df[df["SEX"].isin(["f", "m"])]

    # Convert target and sensitive attribute
    df["Default"] = df["BAD"].map({"BAD": 0, "GOOD": 1})
    print(df["Default"].unique())
    df["Sex"] = df["SEX"].map({"f": 1, "m": 0})
    print(df["Sex"].unique())
    df.drop(columns=["BAD", "SEX"], inplace=True)

    # Define high-cardinality categorical features
    high_card_cols = ["STATE_OF_BIRTH", "RESIDENTIAL_STATE", "PROFESSIONAL_STATE", "COMPANY", "PROFESSION_CODE", "MATE_PROFESSION_CODE"]
    all_cat_cols = df.select_dtypes(include='object').columns.tolist()
    low_card_cat = list(set(all_cat_cols) - set(high_card_cols))

    if mode == 'onehot':
        # One-hot encode low-cardinality features
        df = pd.get_dummies(df, columns=low_card_cat, drop_first=True, dtype=float)

        # Label encode high-cardinality features
        for col in high_card_cols:
            if col in df.columns:
                df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    elif mode == 'label_encoding':
        # Label encode all categorical features
        for col in all_cat_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    else:
        raise ValueError("Invalid mode. Use 'onehot' or 'label_encoding'.")

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Separate inputs and scale
    y = df["Default"].values
    s = df["Sex"].values
    X = df.drop(columns=["Default", "Sex"])

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, s
