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

    plt.xlabel("True label")
    plt.ylabel("Predicted label")
    plt.title("Confusion matrix")
    plt.show()
    # plt.savefig(out)


random.seed(42)
np.random.seed(42)

do_confusion = False
# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\tosefta_dataset.json"))

do_stemmed=True
if do_stemmed:
    def clean_stemmed(text):
        t1 =  text.replace("_ ", "").replace(" _", "")
        t2 = t1.replace(" ו ", " ").replace(" ה ", " ").replace(" ש ", " ").replace(" מ ", " ").replace(" ב ", " ")
        return t2
    # df = pd.read_json(r"dataset_stemmed.json")
    df = pd.read_json(r"dataset_stemmed_dicta.json")
    df.text = df.text.apply(clean_stemmed)
    a=5
a = 5
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

def get_names(text):
    li = []
    for name in names:
        if name in text:
            li += [name.replace(" ", "_")]
    result = " ".join(li)
    return result

def unite_tractates(name):
    if name.startswith("bava"):
        return "nezikin"
    if name == "sanhedrin" or name == "makkot":
        return "sanhedrin_makkot"
    if name.startswith("keilim"):
        return "keilim"
    return name

do_filter = True
if do_filter:
    df = df[df.tractate != "avot"]

do_unite = True
if do_unite:
    df["tractate"] = df.tractate.apply(unite_tractates)

do_balance = True
if do_balance:
    df = get_balanced_dataset(df)


kf = KFold(n_splits=10, shuffle=True)
df["y_pred"] = 0
df["prob_y"] = 0
df["prob_y_pred"] = 0
df["fold"] = 0

do_name = False
if do_name:
    with open("names.txt", "r") as f:
        names = f.readlines()
    names = [name.strip() for name in names]
    names = list(set(names))
    df.text = df.text.apply(get_names)
    # names = [name.replace(" ", "_") for name in names]

def basic_tokenize(text):
    li = text.split()
    li = [re.sub("[., ]", "", word) for word in li]
    return li

# vec = CountVectorizer(tokenizer=basic_tokenize, max_features=100)
# vec = CountVectorizer(tokenizer=basic_tokenize)
vec = CountVectorizer(tokenizer=basic_tokenize, max_features=1000, ngram_range=(3,3))
# vec = CountVectorizer(tokenizer=basic_tokenize, ngram_range=(1,3))
# vec = TfidfVectorizer()

vec.fit(df.text)
joblib.dump(vec, "vectorizer.pkl")

# ch2_df = pd.DataFrame(columns=["feature", "chi2", "p_value"])
# ch2, p_values = chi2(vec.transform(df.text), df.seder)
#
# # ch2_df.feature = vec.vocabulary_
# # ch2_df.feature = vec.inverse_transform(list(range(100)))[0]
# ch2_df.feature = sorted(vec.vocabulary_)
# ch2_df.chi2 = ch2
# ch2_df.p_value = p_values
# ch2_df.to_excel("chi2_top_100.xlsx")


i=0
test_acc = []
for train_idx, test_idx in kf.split(df):

# for tractate in df["tractate"].unique():
#     train_idx = df[df.tractate != tractate].index
#     test_idx = df[df.tractate == tractate].index

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    X_train, y_train = train_df.text.to_numpy(), train_df.seder.to_numpy()
    X_test, y_test = test_df.text.to_numpy(), test_df.seder.to_numpy()

    # model = LogisticRegression(fit_intercept=False)
    model = LogisticRegression(fit_intercept=True)
    # model = RandomForestClassifier()
    # model = XGBClassifier()
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
# explanation_df.to_excel("mishna_weights_nikkud.xlsx")
explanation_df.to_excel("mishna_weights_top_1000.xlsx")
# explanation_df.to_csv("mishna_weights_nikkud.csv")
# df.to_csv("mishna_preds.csv", encoding="utf8")
df.to_excel("mishna_preds.xlsx")

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
df_cm.to_csv("conf_leave_one_out_tractate.csv")


df_cm_seder = pd.DataFrame(columns=["seder", "zraim", "moed", "nashim", "nezikin", "kodashim", "taharot", "wrong", "total", "ratio"])
seder_list = list(df["seder"].unique())
for i, seder in enumerate(seder_list):
    results = [len(df[(df["seder"]==seder) & (df["y_pred"]==j+1)]) for j in range(6)]
    total = len(df[(df["seder"]==seder)])
    wrong = len(df[(df["seder"]==seder) & (df["y_pred"]!=df["seder"])])
    ratio = wrong/total
    df_cm_seder.loc[len(df_cm_seder.index)+1] = [seder] + results + [wrong, total, ratio]
    for j in range(6):
        cm2[i, j] = len(df[(df["seder"]==seder) & (df["y_pred"]==j+1)])
df_cm_seder.to_csv("conf_leave_one_out_seder.csv")



# plot confusion matrices
#TODO: parameter order? train?
do_confusion = True
if do_confusion:
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=True)
    plot_confusion(preds=df["y_pred"], y=df["seder"], do_normalize=False)
    # plot_confusion(cm=cm2, yticklabels=tractate_list)

"""
Logistic
Average acc: 0.735


stemmed max 100
Average acc: 0.487

stemmed
Average acc: 0.687

non stemmed
Average acc: 0.715

1,2
Average acc: 0.722

1,3
Average acc: 0.713


unite tractates and remove avot
Average acc: 0.721

tosefta:
Average acc: 0.718

"""