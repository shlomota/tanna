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
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sn
from bidi.algorithm import get_display
from sklearn.model_selection import KFold
import os
import re

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
SEDER_LIST = ["זרעים", "מועד", "נשים", "נזיקין", "קודשים", "טהרות"]
SEDER_LIST = [get_display(name) for name in SEDER_LIST]


def plot_confusion(preds=None, y=None, cm=None, do_normalize=False, xticklabels=SEDER_LIST, yticklabels=SEDER_LIST):
    if do_normalize:
        if cm is None:
            cm = confusion_matrix(preds, y, normalize="true")
        out = "cm_norm.png"
        cm2 = cm.copy()
        sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".3f", xticklabels=xticklabels, yticklabels=yticklabels)
    else:
        if cm is None:
            cm = confusion_matrix(preds, y)
        out = "cm_abs.png"
        cm2 = cm.copy()
        sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="g", xticklabels=xticklabels, yticklabels=yticklabels)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.show()
    # plt.savefig(out)


random.seed(42)
np.random.seed(42)

do_confusion = False
df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
# df = df.sample(frac=1)

def get_balanced_dataset(df):
    #balance the df
    counts = df.seder.value_counts()
    min_count = min(counts)
    dfs = [[]] * 6
    for i in range(6):
        dfs[i] = df[df.seder == i+1].sample(n=min_count)
    df = pd.concat(dfs, axis=0)
    df = df.reset_index()
    return df


kf = KFold(n_splits=10
           , shuffle=True)
df["y_pred"] = 0
df["prob_y"] = 0
df["prob_y_pred"] = 0
df["fold"] = 0


def basic_tokenize(text):
    li = text.split()
    li = [re.sub("[., ]", "", word) for word in li]
    return li

vec = CountVectorizer(tokenizer=basic_tokenize)
# vec = TfidfVectorizer()
vec.fit(df.text)
joblib.dump(vec, "vectorizer.pkl")

le = LabelEncoder()
le.fit(df["tractate"])
df["tractate_num"] = le.transform(df.tractate)

i=0
for train_idx, test_idx in kf.split(df):

# for tractate in df["tractate"].unique():
#     train_idx = df[df.tractate != tractate].index
#     test_idx = df[df.tractate == tractate].index

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train, y_train = train_df.text.to_numpy(), train_df.tractate_num.to_numpy()
    X_test, y_test = test_df.text.to_numpy(), test_df.tractate_num.to_numpy()

    model = LogisticRegression()
    model.fit(vec.transform(X_train), y_train)

    preds = model.predict(vec.transform(X_test))
    probs = model.predict_proba(vec.transform(X_test))
    df.loc[test_idx, "y_pred"] = le.inverse_transform(preds)
    df.loc[test_idx, "prob_y"] = probs[range(len(probs)), y_test]
    df.loc[test_idx, "prob_y_pred"] = probs[range(len(probs)), preds]
    df.loc[test_idx, "fold"] = i
    print("Test acc: %.3f" % accuracy_score(y_test, preds))
    # joblib.dump(model, "mishna_%d.pkl" % (i))


    i += 1
    break #TODO: remove

def a(x):
    if "<BIAS>" == x:
        return 1
    return occurences[0, vec.vocabulary_[x]]
#
# df.to_excel("mishna_mishna_preds.xlsx")
#
# cm = confusion_matrix(df["y_pred"], df["tractate"])
# res = df.groupby(["y_pred", "tractate"], axis=0).count().reset_index()
# res[["tractate", "y_pred", "text"]].to_excel("cm_tractates.xlsx")

explanation_df = eli5.explain_weights_df(model, vec=vec)
occurences = vec.transform([df.text.str.cat()])
explanation_df = explanation_df[explanation_df.weight.abs() > 0.5]
# explanation_df["count"] = explanation_df["feature"].apply(lambda x: occurences[vec.vocabulary_[x]])
explanation_df["count"] = explanation_df["feature"].apply(a)
explanation_df.to_excel("mishna_weights_no_nikkud_tractate.xlsx")
# explanation_df.to_excel("mishna_weights_nikkud.xlsx")
# explanation_df.to_csv("mishna_weights_nikkud.csv")
# df.to_csv("mishna_preds.csv", encoding="utf8")

#res = df.groupby(["y_pred", "tractate"], axis=0).count().reset_index()


cm2 = np.zeros([63, 63])

df_cm = pd.DataFrame(columns=["tractate", "zraim", "moed", "nashim", "nezikin", "kodashim", "taharot", "wrong", "total", "ratio"])
tractate_list = list(df["tractate"].unique())
for i, tractate in enumerate(tractate_list):
    results = [len(df[(df["tractate"]==tractate) & (df["y_pred"]==j+1)]) for j in range(6)]
    total = len(df[(df["tractate"]==tractate)])
    wrong = len(df[(df["tractate"]==tractate) & (df["y_pred"]!=df["seder"])])
    ratio = wrong/total
    df_cm.loc[len(df_cm.index)+1] = [tractate] + results + [wrong, total, ratio]
    for j in range(6):
        cm2[i, j] = len(df[(df["tractate"]==tractate) & (df["y_pred"]==j+1)])
df_cm.to_csv("conf_leave_one_out.csv")



# plot confusion matrices
#TODO: parameter order? train?
do_confusion = True
if do_confusion:
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=True)
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=False)
    # plot_confusion(cm=cm2, yticklabels=tractate_list)

