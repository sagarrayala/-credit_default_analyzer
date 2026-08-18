import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fetch_credit_card_data(file_path):
    """
    Load the UCI Credit Card Default dataset from an Excel file.

    The dataset has 25 columns (23 features + 1 ID + 1 target).
    This function maps each column to a custom financial term.

    Args:
        file_path (str): Path to the Excel file (e.g., 'data/credit_card_default.xls')

    Returns:
        pandas DataFrame with renamed columns and cleaned values.

    Column Mapping:
        Original Name         -> Custom Name
        ---------------------->------------------------
        ID                    -> client_unique_id
        LIMIT_BAL             -> credit_line_amount
        SEX                   -> gender_code
        EDUCATION             -> edu_level_code
        MARRIAGE              -> marital_status_code
        AGE                   -> age_in_years
        PAY_0                 -> payment_status_sep
        PAY_2                 -> payment_status_aug
        PAY_3                 -> payment_status_jul
        PAY_4                 -> payment_status_jun
        PAY_5                 -> payment_status_may
        PAY_6                 -> payment_status_apr
        BILL_AMT1             -> bill_amount_sep
        BILL_AMT2             -> bill_amount_aug
        BILL_AMT3             -> bill_amount_jul
        BILL_AMT4             -> bill_amount_jun
        BILL_AMT5             -> bill_amount_may
        BILL_AMT6             -> bill_amount_apr
        PAY_AMT1              -> pay_amount_sep
        PAY_AMT2              -> pay_amount_aug
        PAY_AMT3              -> pay_amount_jul
        PAY_AMT4              -> pay_amount_jun
        PAY_AMT5              -> pay_amount_may
        PAY_AMT6              -> pay_amount_apr
        default payment next month -> default_flag
    """
    # Read the Excel file. header=1 means use the second row as column names
    # (the first row in the file contains a duplicate header)
    df = pd.read_excel(file_path, header=1)

    # Define the custom column name mapping
    rename_scheme = {
        'ID': 'client_unique_id',
        'LIMIT_BAL': 'credit_line_amount',
        'SEX': 'gender_code',
        'EDUCATION': 'edu_level_code',
        'MARRIAGE': 'marital_status_code',
        'AGE': 'age_in_years',
        'PAY_0': 'payment_status_sep',  # September 2005
        'PAY_2': 'payment_status_aug',  # August 2005
        'PAY_3': 'payment_status_jul',  # July 2005
        'PAY_4': 'payment_status_jun',  # June 2005
        'PAY_5': 'payment_status_may',  # May 2005
        'PAY_6': 'payment_status_apr',  # April 2005
        'BILL_AMT1': 'bill_amount_sep',
        'BILL_AMT2': 'bill_amount_aug',
        'BILL_AMT3': 'bill_amount_jul',
        'BILL_AMT4': 'bill_amount_jun',
        'BILL_AMT5': 'bill_amount_may',
        'BILL_AMT6': 'bill_amount_apr',
        'PAY_AMT1': 'pay_amount_sep',
        'PAY_AMT2': 'pay_amount_aug',
        'PAY_AMT3': 'pay_amount_jul',
        'PAY_AMT4': 'pay_amount_jun',
        'PAY_AMT5': 'pay_amount_may',
        'PAY_AMT6': 'pay_amount_apr',
        'default payment next month': 'default_flag'
    }

    # Apply the column rename
    df = df.rename(columns=rename_scheme)

    # Clean dirty categorical values
    # According to UCI documentation, EDUCATION has values: 1,2,3,4 but there are 0,5,6
    # Mapping 0,5,6 to 4 (meaning 'Others')
    df['edu_level_code'] = df['edu_level_code'].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE has values: 1,2,3 but there is 0. Map 0 to 3 (meaning 'Others')
    df['marital_status_code'] = df['marital_status_code'].replace({0: 3})

    return df


def engineer_risk_features(df):
    """
    Create custom derived features that demonstrate financial domain expertise.

    These features are NOT present in the original dataset. They are your own
    contribution, making your project unique.

    New features created:
        1. avg_monthly_bill          -> Mean of the 6 monthly bill amounts
        2. credit_utilization_ratio  -> Average bill / credit line (a key risk metric)
        3. payment_volatility        -> Standard deviation of payment amounts
        4. delinquency_trend         -> Latest payment status - earliest payment status

    Args:
        df (pandas DataFrame): Data with original columns (after renaming)

    Returns:
        pandas DataFrame: Original data PLUS the new engineered features
    """
    # Create a copy to avoid modifying the original
    df = df.copy()

    # Feature 1: Average monthly bill over the 6-month period
    bill_cols = [
        'bill_amount_sep', 'bill_amount_aug', 'bill_amount_jul',
        'bill_amount_jun', 'bill_amount_may', 'bill_amount_apr'
    ]
    df['avg_monthly_bill'] = df[bill_cols].mean(axis=1)

    # Feature 2: Credit utilization ratio (avg bill / credit limit)
    # Adding 1 in denominator to avoid division by zero
    df['credit_utilization_ratio'] = df['avg_monthly_bill'] / (df['credit_line_amount'] + 1)

    # Feature 3: Payment volatility (std deviation of 6 payment amounts)
    pay_cols = [
        'pay_amount_sep', 'pay_amount_aug', 'pay_amount_jul',
        'pay_amount_jun', 'pay_amount_may', 'pay_amount_apr'
    ]
    df['payment_volatility'] = df[pay_cols].std(axis=1)

    # Feature 4: Delinquency trend (positive means worsening, negative means improving)
    # Compare latest payment status (Sep) to earliest (Apr)
    df['delinquency_trend'] = df['payment_status_sep'] - df['payment_status_apr']

    return df


def prepare_model_data(df, target_col='default_flag', test_size=0.2):
    """
    Split data into train/test sets and scale numerical features.

    This function:
        1. Separates features (X) from target (y)
        2. Performs a stratified train-test split (maintains class balance)
        3. Scales numerical features using StandardScaler
        4. Leaves categorical features un-scaled (e.g., gender_code, edu_level_code)

    Args:
        df (pandas DataFrame): Data with target and features (after engineering)
        target_col (str): Name of the target column (default: 'default_flag')
        test_size (float): Proportion for test set (default: 0.2)

    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
            - X_train: Scaled training features
            - X_test: Scaled test features
            - y_train: Training target labels
            - y_test: Test target labels
            - scaler: Fitted StandardScaler (for later use in inference)
    """
    # Drop the client ID (not a feature) and separate target
    X = df.drop(columns=[target_col, 'client_unique_id'])
    y = df[target_col]

    # Stratified split to preserve the 22% default rate in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=2026, stratify=y
    )

    # Identify which columns are numerical (exclude categorical ones)
    categorical_cols = ['gender_code', 'edu_level_code', 'marital_status_code']
    num_cols = [col for col in X.select_dtypes(include=[np.number]).columns.tolist()
                if col not in categorical_cols]

    # Scale only numerical features
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
