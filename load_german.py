import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_german(filepath, mode='onehot'):
    columns = [
        "Status", "Duration", "Credit_History", "Purpose", "Credit_Amount",
        "Savings", "Employment", "Installment_Rate", "Personal_Status_Sex",
        "Other_Debtors", "Residence_Since", "Property", "Age", "Other_Installment_Plans",
        "Housing", "Existing_Credits", "Job", "Liable", "Telephone", "Foreign_Worker", "Default"
    ]

    df = pd.read_csv(filepath, delimiter=' ', names=columns)

    # Binary target and gender
    df["Y"] = df["Default"].apply(lambda x: 1 if x == 1 else 0)
    df["sex"] = df["Personal_Status_Sex"].map(lambda x: 1 if x in ["A91", "A93", "A94"] else 0)
    df.drop(["Personal_Status_Sex", "Default"], axis=1, inplace=True)

    # Separate label and sensitive attribute
    y = df["Y"].values
    s = df["sex"].values

    if mode == 'onehot':
        df = pd.get_dummies(df, dtype=float, drop_first=True)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(df.drop(columns=["Y", "sex"]))

    elif mode == 'label_encoding':
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        le_dict = {}

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le  # in case you want to inverse later or need vocab sizes

        num_cols = df.select_dtypes(include=[np.number]).columns.difference(["Y", "sex"])
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        X = df.drop(columns=["Y", "sex"]).values

    else:
        raise ValueError("mode must be either 'onehot' or 'label_encoding'")

    return X, y, s
