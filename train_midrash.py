import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import eli5
import random
import joblib
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sn
from bidi.algorithm import get_display
from sklearn.model_selection import KFold
import os

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"


random.seed(42)
np.random.seed(42)

do_confusion = False
df = pd.read_json(os.path.join(BASE_PATH, r"midrash\dataset.json"))
# df = df.sample(frac=1)

kf = KFold(n_splits=10, shuffle=True)
df["y_pred"] = 0
df["prob_y"] = 0
df["prob_y_pred"] = 0

for train_idx, test_idx in kf.split(df):
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train, y_train = train_df.text.to_numpy(), train_df.source.to_numpy()
    X_test, y_test = test_df.text.to_numpy(), test_df.source.to_numpy()

    # vec = CountVectorizer(ngram_range=(1,3))
    vec = CountVectorizer()
    # vec = TfidfVectorizer()
    vec.fit(df.text)

    model = LogisticRegression()
    model.fit(vec.transform(X_train), y_train)

    train_preds = model.predict(vec.transform(X_train))
    preds = model.predict(vec.transform(X_test))
    probs = model.predict_proba(vec.transform(X_test))
    df.loc[test_idx, "y_pred"] = preds
    df.loc[test_idx, "prob_y"] = probs[range(len(probs)), y_test-1]
    df.loc[test_idx, "prob_y_pred"] = probs[range(len(probs)), preds-1]


    print("Train acc: %.3f" % accuracy_score(y_train, train_preds))
    print("Test acc: %.3f" % accuracy_score(y_test, preds))
    print(sum(y_test==1) / len(y_test))

    try:
        weights = eli5.explain_weights_df(model, vec=vec)
    except:
        weights = None

    weights.to_csv("midrash_weights.csv", encoding="utf8")
    break
