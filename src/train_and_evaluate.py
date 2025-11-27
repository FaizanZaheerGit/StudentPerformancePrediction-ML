import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42


def train_and_evaluate(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "DecisionTree": Pipeline([
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
        ]),
        "Perceptron": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", Perceptron(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                random_state=RANDOM_STATE
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                learning_rate_init=0.001,
                solver="adam",
                max_iter=3000,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in models.items():
        print(f"\nTraining {name} ...")
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred)
        conf_mat = confusion_matrix(y_test, y_pred, labels=np.unique(y_train))

        results[name] = {
            "pipeline": pipe,
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "test_acc": float(acc_test),
            "report": clf_report,
            "confusion_matrix": conf_mat
        }

        print(f"=== {name} ===")
        print(f"CV accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
        print(f"Test accuracy: {acc_test:.3f}")
        print("Classification Report:\n", clf_report)

    return results, X_train, X_test, y_train, y_test
