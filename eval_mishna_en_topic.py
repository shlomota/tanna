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
import re

# BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
BASE_PATH = r"C:\Users\soki\Documents\TAU\Thesis\Mishnah"
SEDER_LIST = ["zeraim", "moed", "nashim", "nezikin", "kodashim", "tahorot"]
# SEDER_LIST = [get_display(name) for name in SEDER_LIST]


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
# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
df = pd.read_json(os.path.join(BASE_PATH, r"dataset_en.json"))
df_topic = pd.read_csv(os.path.join(BASE_PATH, r"topics.csv"))
df["topic_name"] = df_topic.topic_name
df["topic"] = df_topic.topic

tractate_list = list(df["tractate"].unique())
tractate_to_index = {v:i for i, v in enumerate(tractate_list)}
df["tractate_num"] = df.tractate.apply(lambda x: tractate_to_index[x])
# df = df.sample(frac=1)

def get_balanced_dataset(df):
    #balance the df
    counts = df.tractate.value_counts()
    min_count = min(counts)
    dfs = [[]] * len(counts)
    dfs = []
    for tractate in tractate_list:
        dfs += [df[df.tractate == tractate].sample(n=min_count)]

    # for i in range(len(counts)):
    #     dfs[i] = df[df.seder == i+1].sample(n=min_count)
    df = pd.concat(dfs, axis=0)
    df = df.reset_index()
    return df


df = get_balanced_dataset(df)
kf = KFold(n_splits=10, shuffle=True)
df["y_pred"] = 0
df["prob_y"] = 0
df["prob_y_pred"] = 0
df["fold"] = 0


def basic_tokenize(text):
    li = text.split()
    li = [re.sub("[., ]", "", word) for word in li]
    return li

vec = CountVectorizer(tokenizer=basic_tokenize, lowercase=False)
# vec = CountVectorizer(tokenizer=basic_tokenize, lowercase=True)
# vec = TfidfVectorizer()
# df.text = df.text.str.lower()
# vec.fit(df.text)
vec.fit(df.topic_name)
joblib.dump(vec, "vectorizer.pkl")

i=0
test_acc = []
for train_idx, test_idx in kf.split(df):

# for tractate in df["tractate"].unique():
#     train_idx = df[df.tractate != tractate].index
#     test_idx = df[df.tractate == tractate].index

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train, y_train = train_df.topic_name.to_numpy(), train_df.tractate_num.to_numpy()
    # X_train, y_train = train_df.text.to_numpy(), train_df.seder.to_numpy()
    X_test, y_test = test_df.topic_name.to_numpy(), test_df.tractate_num.to_numpy()
    # X_test, y_test = test_df.text.to_numpy(), test_df.seder.to_numpy()

    model = LogisticRegression()
    model.fit(vec.transform(X_train), y_train)

    preds = model.predict(vec.transform(X_test))
    probs = model.predict_proba(vec.transform(X_test))
    df.loc[test_idx, "y_pred"] = preds
    df.loc[test_idx, "prob_y"] = probs[range(len(probs)), y_test-1]
    df.loc[test_idx, "prob_y_pred"] = probs[range(len(probs)), preds-1]
    print("Test acc: %.3f" % accuracy_score(y_test, preds))
    # joblib.dump(model, "mishna_%d.pkl" % (i))
    test_acc += [accuracy_score(y_test, preds)]


    i += 1
    # break #TODO: remove

print("Average acc: %.3f" % np.average(test_acc))

def a(x):
    if "<BIAS>" == x:
        return 1
    return occurences[0, vec.vocabulary_[x]]

explanation_df = eli5.explain_weights_df(model, vec=vec)
occurences = vec.transform([df.text.str.cat()])
# explanation_df["count"] = explanation_df["feature"].apply(lambda x: occurences[vec.vocabulary_[x]])
explanation_df["count"] = explanation_df["feature"].apply(a)
explanation_df["name"] = explanation_df["target"].apply(lambda x: tractate_list[x])
# explanation_df.to_excel("mishna_weights_nikkud.xlsx")
explanation_df.to_excel("mishna_weights_en_topic.xlsx")
# explanation_df.to_csv("mishna_weights_nikkud.csv")
# df.to_csv("mishna_preds_en.csv", encoding="utf8")

cm = confusion_matrix(df["y_pred"], df["seder"])
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
# df_cm.to_csv("conf_leave_one_out_en.csv")



# plot confusion matrices
#TODO: parameter order? train?
do_confusion = True
if do_confusion:
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=True)
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=False)
    # plot_confusion(cm=cm2, yticklabels=tractate_list)

"""
Logistic
Average acc: 0.773
"""