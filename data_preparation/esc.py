import pandas as pd
import os

# Base directory
base_dir = "data1/ESCAPE_DS"
output_dir = "data1"
# Input files
files = ["Fold1.csv", "Fold2.csv", "Test.csv"]

# Column names
cols = ["sequence", "id", "l1", "l2", "l3", "l4", "l5"]

# List to store all inactive data
all_inactive = []

for file in files:
    file_path = os.path.join(base_dir, file)
    
    # Read file
    df = pd.read_csv(file_path, header=None)
    df.columns = cols

    # Remove empty sequences
    df = df[df["sequence"].notna() & (df["sequence"].str.strip() != "")]

    # Convert labels to numeric
    for col in ["l1", "l2", "l3", "l4", "l5"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Filter inactive (all zeros)
    inactive_df = df[df[["l1", "l2", "l3", "l4", "l5"]].sum(axis=1) == 0]

    all_inactive.append(inactive_df)

# Combine all data
final_df = pd.concat(all_inactive, ignore_index=True)

# Save single file
output_path = os.path.join(output_dir, "inactive.csv")
final_df.to_csv(output_path, index=False)

print(f"Total inactive peptides: {len(final_df)}")
print(f"Saved to: {output_path}")