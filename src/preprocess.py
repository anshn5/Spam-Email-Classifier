import os
import pandas as pd

# Define file paths
RAW_FILE = "data/spambase/spambase.data"
OUTPUT_FILE = "data/spam_dataset.csv"

# Create column names for 57 features and 1 target label
columns = [f"feature_{i}" for i in range(1, 58)] + ["label"]

print("Loading email dataset...")

# Load the dataset
df = pd.read_csv(
    RAW_FILE,
    header=None,
    names=columns
)

# Convert numeric labels into readable categories
df["label"] = df["label"].map({
    0: "Not Spam",
    1: "Spam"
})

# Remove any rows with missing values
df.dropna(inplace=True)

# Save the processed dataset
df.to_csv(OUTPUT_FILE, index=False)

# Display dataset information
print("\nDataset processed successfully!")
print("Total emails:", len(df))
print("Spam emails:", (df["label"] == "Spam").sum())
print("Not spam emails:", (df["label"] == "Not Spam").sum())
print("Saved to:", OUTPUT_FILE)

print("\nFirst five rows:")
print(df.head())