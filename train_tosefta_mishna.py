import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import eli5
import random

random.seed(42)
np.random.seed(42)

df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\tosefta_dataset.json")

train_df, test_df = train_test_split(df, test_size=0.2)

X_train, y_train = train_df.text.to_numpy(), train_df.seder.to_numpy()
X_test, y_test = test_df.text.to_numpy(), test_df.seder.to_numpy()

# vec = TfidfVectorizer()
vec = CountVectorizer()
# vec = TfidfVectorizer(ngram_range=(1,2))
vec.fit(df.text)

model = LogisticRegression()
# model = Pipeline([vec, model])
# model = RandomForestClassifier()
# model = XGBClassifier()
# model = Pipeline([("feature_selection", SelectKBest(chi2, k=100)), ("model", model)])
model.fit(vec.transform(X_train), y_train)
# model.fit(np.array(list(map(lambda x: len(x), X_train))).reshape(-1, 1), y_train)

try:
    weights = eli5.explain_weights_df(model, vec=vec)
except:
    weights = None

preds = model.predict(vec.transform(X_train))
# preds = model.predict(np.array(list(map(lambda x: len(x), X_train))).reshape(-1, 1))
print("Train acc: %.3f" % accuracy_score(y_train, preds))

preds = model.predict(vec.transform(X_test))
# preds = model.predict(np.array(list(map(lambda x: len(x), X_test))).reshape(-1, 1))
print("Test acc: %.3f" % accuracy_score(y_test, preds))

weights.to_csv("tosefta_unigram_logistic_weights2.csv", encoding="utf8")

print("FPs: %s" % len(np.where(np.logical_and(preds==2, y_test==1))[0]))
print("FNs: %s" % len(np.where(np.logical_and(preds==1, y_test==2))[0]))

#explain prediction
for idx in range(10):
    print(X_test[idx])
    print(model.predict(vec.transform([X_test[idx]]))[0])
    print(y_test[idx])
    print("="*20)
    ex = eli5.explain_prediction_df(model, X_test[idx], vec=vec)
    a=5

"""
Logistic:
Unigram:
Train acc: 1.000
Test acc: 0.892

Select 100
Train acc: 0.870
Test acc: 0.881


Uniform data
Unigram:
Train acc: 0.999
Test acc: 0.831

Select 100
Train acc: 0.770
Test acc: 0.794


Sentence:
Train acc: 0.971
Test acc: 0.888

Just length:
Train acc: 0.819
Test acc: 0.818
"""