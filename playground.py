import pandas as pd
import os
import joblib
import eli5
BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
df = pd.read_json(os.path.join(BASE_PATH, r"midrash\dataset.json"))
# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
# df = pd.read_csv("trigram_logistic_weights.csv", encoding="utf8")
a=5

text = """המקדש בחלקו, בין קדשי קדשים בין קדשים קלים, אינה מקדשת. במעשר שני, בין שוגג בין מזיד, לא קדש, דברי רבי מאיר. רבי יהודה אומר, בשוגג לא קדש, במזיד קדש. ובהקדש, במזיד קדש ובשוגג לא קדש, דברי רבי מאיר. רבי יהודה אומר, בשוגג קדש, במזיד לא קדש """
model = joblib.load("log_mishna_clf.pkl")
vec = joblib.load("output/mishna_vec.pkl")

explanation_df = eli5.explain_prediction_df(model, text, vec=vec)
print(model.predict_proba(vec.transform([text])))