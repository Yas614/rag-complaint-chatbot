import pandas as pd

df = pd.read_csv("data/raw/complaints.csv")

print(df["Product"].value_counts())
df["word_count"] = df["Consumer complaint narrative"].fillna("").apply(
    lambda x: len(x.split())
)
import matplotlib.pyplot as plt

plt.hist(df["word_count"], bins=50)
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.show()
print(df["Consumer complaint narrative"].isna().sum())
products = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]
filtered = df[
    df["Product"].isin(products)
]
filtered = filtered[
    filtered["Consumer complaint narrative"].notna()
]
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
filtered["cleaned_text"] = (
    filtered["Consumer complaint narrative"]
    .apply(clean_text)
)
filtered.to_csv(
    "data/processed/filtered_complaints.csv",
    index=False
)