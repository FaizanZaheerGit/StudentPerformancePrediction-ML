import os
from joblib import dump, load

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model_artifacts(pipeline, label_encoder, model_name="best_model.joblib"):
    save_path = os.path.join(MODELS_DIR, model_name)
    try:
        dump({"pipeline": pipeline, "label_encoder": label_encoder}, save_path)
        print(f"Saved model artifacts to: {save_path}")
    except Exception as e:
        print(f"Error saving model artifacts: {e}")


def load_model_artifacts(model_name="best_model.joblib"):

    load_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model artifact not found: {load_path}")
    obj = load(load_path)
    return obj["pipeline"], obj["label_encoder"]


def predict_from_input(pipeline, numeric_features, le_target):

    import numpy as np

    print("\nEnter values for features (press Enter to use 0):")
    vals = []
    for feat in numeric_features:
        raw = input(f"{feat}: ").strip()
        if raw == "":
            raw = "0"
        try:
            v = float(raw)
        except Exception:
            print("Invalid input; using 0.")
            v = 0.0
        vals.append(v)

    arr = np.array(vals).reshape(1, -1)
    pred = pipeline.predict(arr)
    label = le_target.inverse_transform(pred)
    print(f"\nPredicted Class: {label[0]}")
