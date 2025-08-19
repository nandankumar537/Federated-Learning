import pandas as pd

cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
        "thalach","exang","oldpeak","slope","ca","thal","target"]

# Open in binary, replace NULs, decode, then let pandas read from string buffer
with open("cleveland.data", "rb") as f:
    raw = f.read().replace(b"\x00", b"")  # strip NULs

from io import StringIO
df = pd.read_csv(StringIO(raw.decode("utf-8", errors="replace")),
                 header=None, names=cols, sep=",", na_values=["?"], engine="c")

# enforce numeric
for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df.to_csv("cleveland.csv", index=False, encoding="utf-8")
print("Saved cleveland.csv", df.shape)
