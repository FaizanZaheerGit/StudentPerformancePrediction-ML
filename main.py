from src.load_data import load_data, save_data
from src.preprocessing import prepare_features
from src.train_and_evaluate import train_and_evaluate
from src.visualization import menu_plots
from src.inference import save_model_artifacts, predict_from_input, load_model_artifacts
import os

DATA_PATH = "data/raw/AI-Data.csv"


def main():

    df = load_data(DATA_PATH)
    if df is None:
        return

    menu_plots(df)

    print("\nPreparing features...")
    X, y, numeric_features, le_target = prepare_features(df, target_col="Class")
    print(f"Prepared features: {numeric_features}")

    print("\nTraining and evaluating models...")
    results, X_train, X_test, y_train, y_test = train_and_evaluate(X, y)
    print("Training complete.\n")

    save_data(df)

    best_name = max(results, key=lambda k: results[k]["test_acc"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"Best model: {best_name} (test_acc={results[best_name]['test_acc']:.3f})")
    save_model_artifacts(best_pipeline, le_target, model_name=f"{best_name}.joblib")

    if input("\nDo you want to test a custom input interactively? (y/n): ").lower().startswith("y"):

        pipeline, le = load_model_artifacts(f"{best_name}.joblib")
        predict_from_input(pipeline, numeric_features, le)

    print("\nAll done.")


if __name__ == "__main__":
    main()
