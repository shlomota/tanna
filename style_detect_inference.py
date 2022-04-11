import joblib
import preprocess_style
import glob
import os
import pandas as pd
mode = "normal"

model = joblib.load(f"style_{mode}_logistic.pkl")
vec = joblib.load(f"style_{mode}_logistic_vec.pkl")

path = r"C:\Users\soki\Documents\TAU\DL\proj\style\inference\Bamidbar Rabbah\Hebrew\merged.txt"
paths = glob.glob(os.path.join(r"C:\Users\soki\Documents\TAU\DL\proj\style\inference", r"*\Hebrew\merged.txt"))
books = [path.split("\\")[-3] for path in paths]
df = pd.DataFrame(columns=books)

for i, path in enumerate(paths):
    print(books[i])
    chunks = preprocess_style.preprocess_single_file(path)
    preds = model.predict(vec.transform(chunks))
    s = pd.Series(preds)
    print(s.value_counts())
    df[books[i]] = s.value_counts(sort=False)
    a = 5


c = 5