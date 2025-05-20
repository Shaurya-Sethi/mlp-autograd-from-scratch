import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


def validate_dataframe(df):
    """
    Checks for required columns and validates types for the input DataFrame.
    """
    # Full column list from the screenshot
    required_cols = [
        "min", "median", "ptp", "rms", "zcr", "var",
        "skew_kurt_ratio", "crest_factor", "shape_factor",
        "skew_kurt_product", "std_min_ratio", "min_ptp_ratio", "label"
    ]

    # 1. Check presence of columns
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the CSV.")

    # 2. Check that numeric columns are numeric
    numeric_cols = [c for c in required_cols if c != "label"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric. Found dtype={df[col].dtype}.")

    # 3. Check that 'label' column is not empty
    if df["label"].isnull().any():
        raise ValueError("Some rows have a null label. Please remove or fill them.")


def load_and_preprocess_data(csv_path):
    """
    Loads and preprocesses data from CSV for training.
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Validate columns
    validate_dataframe(df)

    # Drop unwanted columns
    df = df.drop(columns=["rms", "skew_kurt_ratio"])

    # Drop any remaining NaNs if present
    df = df.dropna()

    # Separate label
    X_df = df.drop(columns=["label"])
    y_df = df["label"]

    # Standard scaling
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X_df)

    # Label encode
    label_encoder = LabelEncoder()
    y_np = label_encoder.fit_transform(y_df)

    return X_np, y_np, label_encoder



def save_model(model, filename="mlp_model.pkl"):
    """
    Save only the model weights (not Tensor objects) to avoid pickle issues.
    """
    model_weights = [p.data for p in model.params]  # Extract `.data` from Tensors
    with open(filename, "wb") as f:
        pickle.dump(model_weights, f)
    print(f"Model saved successfully to {filename}.")


def load_model(model, filename="mlp_model.pkl"):
    """
    Load model weights into an existing model instance.
    """
    with open(filename, "rb") as f:
        model_weights = pickle.load(f)

    for p, loaded_w in zip(model.params, model_weights):
        p.data = loaded_w  # Restore weights

    print(f"Model loaded successfully from {filename}.")