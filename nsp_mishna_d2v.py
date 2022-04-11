import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sn
import os
import re
import torch
from torch import nn
from tqdm import tqdm

tqdm.pandas()

from gensim.models import Word2Vec
import gensim


# BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
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


def unite_tractates(name):
    if name.startswith("bava"):
        return "nezikin"
    if name == "sanhedrin" or name == "makkot":
        return "sanhedrin_makkot"
    if name.startswith("keilim"):
        return "keilim"
    return name


random.seed(42)
np.random.seed(42)



do_confusion = False

if os.path.exists("pairs_df.csv"):
    pairs_df = pd.read_csv("pairs_df.csv", encoding="utf8")

else:
    # BASE_PATH = r"/content/drive/MyDrive/tanna"
    df = pd.read_json("dataset.json")
    # df = pd.read_json(os.path.join(BASE_PATH, r"dataset.json"))

    do_filter = True
    if do_filter:
        df = df[df.tractate != "avot"]

    do_unite = True
    if do_unite:
        df["tractate"] = df.tractate.apply(unite_tractates)

    pairs_df = pd.DataFrame(columns=["text1", "text2", "label"])

    for i in range(len(df) - 1):
        if df.iloc[i]["chapter"] == df.iloc[i+1]["chapter"]:
            pairs_df.loc[len(pairs_df)] = [df.iloc[i].text, df.iloc[i+1].text, 1]
            pairs_df.loc[len(pairs_df)] = [df.iloc[i].text, df.iloc[random.randint(0, len(df) - 1)].text, 0]


model = gensim.models.doc2vec.Doc2Vec.load("doc2vec.model")
# model = Word2Vec.load("word2vec.model")
def cos_sim(a, b):
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return similarity

def tokenize(text):
    return text.replace(",","").replace(".","").split()

def get_sent_emb(sent):
    words = tokenize(sent)
    emb = model.infer_vector(words)
    # fwords = list(filter(lambda w: w in model.wv, words))
    # embs = [model.wv[word] for word in fwords]
    # emb = np.average(np.array(embs), axis=0)
    return emb

def sim(row):
    return cos_sim(get_sent_emb(row["text1"]), get_sent_emb(row["text2"]))

preds = pairs_df.progress_apply(sim, axis=1)
a=5
thresh = 0.882
thresh = np.median(preds)
print(thresh)
print(classification_report(pairs_df.label, preds>thresh))

score = roc_auc_score(pairs_df.label.values, preds)
print(score)

tpr, fpr, _ = roc_curve(pairs_df.label.values, preds)
plt.plot(tpr, fpr)
plt.title("ROC curve")
plt.show()

precision, recall, _ = precision_recall_curve(pairs_df.label.values, preds)
plt.plot(precision, recall)
plt.title("precision recall curve")
plt.show()



"""
0.35898514091968536
              precision    recall  f1-score   support

           0       0.69      0.69      0.69      3565
           1       0.69      0.69      0.69      3565

    accuracy                           0.69      7130
   macro avg       0.69      0.69      0.69      7130
weighted avg       0.69      0.69      0.69      7130

0.7576672456424369
"""