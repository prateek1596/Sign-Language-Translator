import pandas as pd

df = pd.read_csv("landmarks.csv")
df = df.dropna()
df.to_csv("landmarks.csv", index=False)

print("Cleaned rows:", len(df))
