import joblib
import pandas as pd

MODEL_FILE = "models/spam_classifier.pkl"
DATA_FILE = "data/spam_dataset.csv"

print("Loading trained spam classifier...")

model = joblib.load(MODEL_FILE)
df = pd.read_csv(DATA_FILE)

# Select a sample email record
print("\nChoose a sample email from the dataset.")
print("Enter a row number from 0 to", len(df) - 1)

try:
    row_number = int(input("Row number: "))

    if row_number < 0 or row_number >= len(df):
        raise ValueError("Row number is out of range.")

    sample = df.iloc[[row_number]]
    actual_label = sample["label"].iloc[0]
    features = sample.drop(columns=["label"])

    prediction = model.predict(features)[0]

    print("\n========== PREDICTION ==========")
    print("Actual label    :", actual_label)
    print("Predicted label :", prediction)
    print("================================")

except ValueError as error:
    print("Invalid input:", error)