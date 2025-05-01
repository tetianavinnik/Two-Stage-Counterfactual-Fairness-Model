import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_card_credit(filepath, mode='onehot'):
    '''Imports and prepares the UCI Default of Credit Card Clients dataset with optional encoding modes.'''

    # Load the data from Excel file
    df = pd.read_excel(filepath, header=1)
    
    # Ensure the dataset has the expected columns
    expected_columns = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
                        'default payment next month']
    
    # Handle different column naming conventions
    if set(expected_columns).issubset(set(df.columns)):
        pass
    elif 'Y' in df.columns:  # Check if columns are named X1, X2, ..., Y
        # Map from X1, X2, etc. to meaningful names
        column_mapping = {
            'X1': 'ID', 'X2': 'LIMIT_BAL', 'X3': 'SEX', 'X4': 'EDUCATION', 'X5': 'MARRIAGE',
            'X6': 'AGE', 'X7': 'PAY_0', 'X8': 'PAY_2', 'X9': 'PAY_3', 'X10': 'PAY_4',
            'X11': 'PAY_5', 'X12': 'PAY_6', 'X13': 'BILL_AMT1', 'X14': 'BILL_AMT2',
            'X15': 'BILL_AMT3', 'X16': 'BILL_AMT4', 'X17': 'BILL_AMT5', 'X18': 'BILL_AMT6',
            'X19': 'PAY_AMT1', 'X20': 'PAY_AMT2', 'X21': 'PAY_AMT3', 'X22': 'PAY_AMT4',
            'X23': 'PAY_AMT5', 'X24': 'PAY_AMT6', 'Y': 'default payment next month'
        }
        # Rename columns that exist in the dataframe
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)

    # Drop rows with missing values in target or SEX
    df = df.dropna(subset=["default payment next month", "SEX"])

    # Convert SEX to binary format (0=male, 1=female) - in original data 1=male, 2=female
    df["Sex"] = df["SEX"].map({1: 0, 2: 1})
    
    # The target is already binary (0=no default, 1=default)
    df["Default"] = df["default payment next month"]
    
    # Features to use
    selected_features = [
        # Demographics
        'Sex', 'Default', 'LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE',
        
        # Payment history (PAY_X): -2=no consumption, -1=paid in full, 0=revolving credit, 1=payment delay for 1 month, etc.
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        
        # Amount of bill statements
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        
        # Amount of previous payments
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    # Keep only columns that exist in the dataframe
    existing_features = [f for f in selected_features if f in df.columns]
    df = df[existing_features].copy()
    
    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Get final X, y, s outputs
    y = df['Default'].astype(int).values  # Target
    s = df['Sex'].astype(int).values      # Protected attribute
    
    # Separate features
    X = df.drop(columns=['Default', 'Sex'])
    
    # Handle categorical features
    if mode == 'onehot':
        # Identify categorical columns
        cat_cols = ['EDUCATION', 'MARRIAGE'] 
        cat_cols = [col for col in cat_cols if col in X.columns]
        
        # One-hot encode categorical features
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=float)
                
    elif mode == 'label_encoding':
        # Label encode categorical features
        cat_cols = ['EDUCATION', 'MARRIAGE']
        cat_cols = [col for col in cat_cols if col in X.columns]
        
        for col in cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    else:
        raise ValueError("Invalid mode. Use 'onehot' or 'label_encoding'.")
    
    # Fill any remaining missing values
    X = X.fillna(X.median(numeric_only=True))
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, s
