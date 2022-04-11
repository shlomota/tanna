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
import matplotlib.pyplot as plt
import seaborn as sn
from bidi.algorithm import get_display

from sklearn.model_selection import KFold
import os
import re


tractate_list = ["זרעים", "מועד", "נשים", "נזיקין", "קודשים", "טהרות", "אחר"]
tractate_list = [get_display(name) for name in tractate_list]


random.seed(42)
np.random.seed(42)

df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\tosefta_dataset.json")

# train_df, test_df = train_test_split(df, test_size=0.2)
#
# X_train, y_train = train_df.text.to_numpy(), train_df.label.to_numpy()
# X_test, y_test = test_df.text.to_numpy(), test_df.label.to_numpy()
X, y = df.text.to_numpy(), df.seder.to_numpy()

# # vec = TfidfVectorizer()
# vec = CountVectorizer()
# # vec = TfidfVectorizer(ngram_range=(1,2))
# vec.fit(df.text)
#
# model = LogisticRegression()
# model = Pipeline([vec, model])
# model = RandomForestClassifier()
# model = XGBClassifier()
# model.fit(vec.transform(X_train), y_train)
model = joblib.load("log_mishna_clf.pkl")
vec = joblib.load("output/mishna_vec.pkl")

try:
    weights = eli5.explain_weights_df(model, vec=vec)
except:
    weights = None

# preds = model.predict(vec.transform(X_train))
# print("Train acc: %.3f" % accuracy_score(y_train, preds))

preds = model.predict(vec.transform(X))
print("Acc: %.3f" % accuracy_score(y, preds))

# get worst mistakes:
probs = model.predict_proba(vec.transform(X))

df = pd.DataFrame(data=np.array([X, y, preds, probs[np.arange(len(y)), y-1], probs[np.arange(len(y)), preds-1]]).T, columns=["text", "y", "y_pred", "prob_y", "prob_y_pred"])
# df.to_excel("mistakes.xlsx")
a=5

ex = eli5.explain_prediction_df(model, X[1481], vec=vec)
a=5

cm = confusion_matrix(preds, y, normalize="true")
cm2 = cm.copy()
# sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".2g", xticklabels=range(1, len(cm2)+1), yticklabels=range(1, len(cm2)+1))
sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".2g", xticklabels=tractate_list, yticklabels=tractate_list)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix normalized by true label")
plt.show()
# plt.savefig("cm_norm.png")

cm = confusion_matrix(preds, y)
cm2 = cm.copy()
sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="g", xticklabels=tractate_list, yticklabels=tractate_list)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix absolute")
plt.show()
# plt.savefig("cm_abs.png")


cm2 = cm.copy()
for i in range(len(cm2)):
    cm2[i,i]=0
sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="g", xticklabels=tractate_list, yticklabels=tractate_list)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix absolute errors only")
plt.show()

# weights.to_csv("bigram_logistic_weights.csv", encoding="utf8")
"""
Logistic:
Unigram:
0.697

Top 1000:


No intercept:


Full Mishna training:
0.713
"""