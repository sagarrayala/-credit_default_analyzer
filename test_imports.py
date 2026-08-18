from src.utils.data_processor import fetch_credit_card_data, engineer_risk_features, prepare_model_data

# Check if the module imports successfully
print("✅ All imports successful!")

# Try loading the dataset (adjust path if needed)
try:
    df = fetch_credit_card_data('data/credit_card_default.xls')
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()[:5]}... (showing first 5)")

    # Try feature engineering
    df_engineered = engineer_risk_features(df)
    print(f"✅ Feature engineering successful! New columns: {df_engineered.columns.tolist()[-4:]}")

    # Try train-test split
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(df_engineered)
    print(f"✅ Train-test split successful!")
    print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"   Default rate in train: {y_train.mean():.2%}")
    print(f"   Default rate in test: {y_test.mean():.2%}")

except FileNotFoundError:
    print("❌ ERROR: Dataset file not found. Make sure 'data/credit_card_default.xls' exists.")
except Exception as e:
    print(f"❌ ERROR: {e}")