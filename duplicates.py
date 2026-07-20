import pandas as pd

df = pd.read_csv("Data/train.csv")

duplicates = df.duplicated(
    subset=["Question", "Answer"]
).sum()

print("Duplicates:", duplicates)