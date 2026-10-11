
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = "data/spam_dataset.csv"
MODEL_DIR = "models"
GRAPH_DIR = "graphs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

print("Loading processed dataset...")
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, pos_label="Spam"
        ),
        "Recall": recall_score(
            y_test, predictions, pos_label="Spam"
        ),
        "F1 Score": f1_score(
            y_test, predictions, pos_label="Spam"
        )
    }

    results.append(metrics)
    trained_models[name] = model

    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1 Score']:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("data/model_comparison.csv", index=False)

# Select the model with the highest F1 score
best_name = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]
best_model = trained_models[best_name]

print(f"\nBest model: {best_name}")

best_predictions = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, best_predictions))

# Save the best model
model_path = os.path.join(MODEL_DIR, "spam_classifier.pkl")
joblib.dump(best_model, model_path)

# Save test predictions
prediction_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": best_predictions
})
prediction_df.to_csv("data/test_predictions.csv", index=False)

# Create confusion matrix
cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["Not Spam", "Spam"]
)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Spam", "Spam"],
    yticklabels=["Not Spam", "Spam"]
)
plt.title(f"Confusion Matrix - {best_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

# Compare model metrics
results_plot = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]
results_plot.plot(kind="bar", figsize=(10, 6))
plt.title("Spam Classifier Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(GRAPH_DIR, "model_comparison.png"),
    dpi=300
)
plt.close()

print("\nTraining completed successfully!")
print("Best model saved to:", model_path)
print("Comparison saved to: data/model_comparison.csv")
print("Graphs saved in:", GRAPH_DIR)
