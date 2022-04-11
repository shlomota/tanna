import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
from dicta_request import get_morph, get_diacriticized, get_stemmed
from morph_parse import parse_num_to_str, parse_num
import pandas as pd
from tqdm import tqdm

np.random.seed(42)

tqdm.pandas()

# label_names = ["MI", "HA", "JT", "BT", "AG", "TAN"]
label_names = [0, 1, 2, 3, 4, 5]
# mode = "diac"
mode = "morph"
mode = "morph_normal"
mode = "morph_and_normal"
mode = "normal"

if mode == "diac":
    df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_diac.json")
elif mode == "morph":
    df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_morph.json")
    # df["text"] = df.text.progress_apply(lambda x: " ".join([parse_num_to_str(y) for y in x.split()]))
    a = 5
elif mode == "morph_normal" or mode == "morph_and_normal":
    df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset.json")
    df2 = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_morph.json")
    df["text2"] = df2.text
else:
    # df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset.json")
    df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_220410.json")
    a = 5

def get_balanced_dataset(df):
    #balance the df
    counts = df.label.value_counts()
    min_count = min(counts)
    dfs = [[]] * len(counts)
    dfs = []
    for label in df.label.unique():
        dfs += [df[df.label == label].sample(n=min_count)]

    # for i in range(len(counts)):
    #     dfs[i] = df[df.seder == i+1].sample(n=min_count)
    df = pd.concat(dfs, axis=0)
    df = df.reset_index()
    return df


df = get_balanced_dataset(df)
#
# df = df.sample(n=100) #TODO: remove
# do_morph = True
# do_diac = False
# do_stem = False
# if do_morph:
#     # chunks_df = chunks_df.parallel_apply(get_morph)
#     chunks = [get_morph(chunk) for chunk in tqdm(chunks)]
# elif do_diac:
#     # chunks_df = chunks_df.parallel_apply(get_diacriticized)
#     chunks = [get_diacriticized(chunk) for chunk in tqdm(chunks)]
# elif do_stem:
#     # chunks_df = chunks_df.parallel_apply(get_diacriticized)
#     chunks = [get_diacriticized(chunk) for chunk in tqdm(chunks)]
# print(df.info())
# print(df.head())
df_train, df_test = train_test_split(df, test_size=0.2)

# vec = TfidfVectorizer()
vec = TfidfVectorizer(ngram_range=(1,3), max_features=10**4)
# vec = CountVectorizer(ngram_range=(1,3), max_features=10**4)
# vec = TfidfVectorizer(ngram_range=(1,3), max_features=10**2)
# vec = TfidfVectorizer(ngram_range=(1,3), max_features=10**3)
# vec = TfidfVectorizer(ngram_range=(1,1), max_features=10**2)
# vec = CountVectorizer(ngram_range=(1,3), max_features=10**3)
# vec = TfidfVectorizer(ngram_range=(1,1), max_features=10**3)
vec.fit(df_train.text)

if mode == "morph_normal":
    vec2 = TfidfVectorizer(ngram_range=(1,3), max_features=10**4)
    vec2.fit(df_train.text2)
    X_train, y_train = df_train[["text", "text2"]], df_train.label
    X_test, y_test = df_test[["text", "text2"]], df_test.label
    model = LogisticRegression(fit_intercept=False)
    X1_train = vec.transform(X_train.text).todense()
    X2_train = vec.transform(X_train.text2).todense()
    X1_test = vec.transform(X_test.text).todense()
    X2_test = vec.transform(X_test.text2).todense()
    model.fit(np.concatenate([X1_train, X2_train], axis=1), y_train)
    train_preds = model.predict(np.concatenate([X1_train, X2_train], axis=1))
    preds = model.predict(np.concatenate([X1_test, X2_test], axis=1))

elif mode == "morph_and_normal":
    X_train, y_train = df_train["text"], df_train.label
    X_test, y_test = df_test["text"], df_test.label
    model = LogisticRegression(fit_intercept=False)
    model.fit(vec.transform(X_train), y_train)
    train_preds = model.predict(vec.transform(X_train))
    preds = model.predict(vec.transform(X_test))

    vec2 = TfidfVectorizer(ngram_range=(1,3), max_features=10**4)
    vec2.fit(df_train.text2)
    X_train2, y_train2 = df_train["text2"], df_train.label
    X_test2, y_test2 = df_test["text2"], df_test.label
    model2 = LogisticRegression(fit_intercept=False)
    model2.fit(vec2.transform(X_train2), y_train2)
    train_preds2 = model2.predict(vec2.transform(X_train2))
    preds2 = model2.predict(vec2.transform(X_test2))



else:
    X_train, y_train = df_train["text"], df_train.label
    X_test, y_test = df_test["text"], df_test.label

    model = LogisticRegression(fit_intercept=False)
    # model = RandomForestClassifier()
    # model = GridSearchCV(
    #     RandomForestClassifier(random_state=17),
    #     param_grid={
    #         'n_estimators': [10, 50, 100, 200],
    #         'max_features': [2, 3, 4, 5],
    #     }
    # )
    model.fit(vec.transform(X_train), y_train)

    train_preds = model.predict(vec.transform(X_train))
    preds = model.predict(vec.transform(X_test))


print("Train acc: %.3f" % accuracy_score(y_train, train_preds))
print("Test acc: %.3f" % accuracy_score(y_test, preds))


joblib.dump(model, f"style_{mode}_logistic.pkl")
joblib.dump(vec, f"style_{mode}_logistic_vec.pkl")
# cm = confusion_matrix(preds, y_test, normalize="true")
cm = confusion_matrix(preds, y_test)
cm2 = cm.copy()
# sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".2g", xticklabels=range(1, len(cm2)+1), yticklabels=range(1, len(cm2)+1))
sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="d", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix normalized by true label")
plt.show()

do_explain = True

output_df = pd.DataFrame(["text", "pred_text", "pred_morph", "label"])
df_test["pred_text"] = preds
# df_test["pred_morph"] = preds2
# df_test.to_excel("style_preds.xlsx")
def get_counts(x):
    if "<BIAS>" == x:
        return 1
    return occurences[0, vec.vocabulary_[x]]

if do_explain:
    explanation_df = eli5.explain_weights_df(model, vec=vec)
    occurences = vec.transform([df.text.str.cat()])
    explanation_df["count"] = explanation_df["feature"].apply(get_counts)
    explanation_df.to_excel(f"style_weights_{mode}.xlsx")

"""
4 word windows:
Train acc: 0.620
Test acc: 0.602

df.label.value_counts()
Out[16]: 
3    463957
2    198402
0    123591
1    105324
4     77867


50 word windows Logistic:
Train acc: 0.941
Test acc: 0.872
50 word windows Logistic ngram 1,3:
Train acc: 0.968
Test acc: 0.856
50 word windows Logistic ngram 1,3 1000:
Train acc: 0.877
Test acc: 0.852
50 word windows Logistic ngram 1,3 10000:
Train acc: 0.934
Test acc: 0.890
50 word windows Logistic ngram 1,2 10000:
Train acc: 0.936
Test acc: 0.886

count 50 word windows Logistic ngram 1,3 10000:
Train acc: 1.000
Test acc: 0.880

50 word windows RF ngram 1,3 1000:
Train acc: 1.000
Test acc: 0.823

logistic morph_normal
Train acc: 0.936
Test acc: 0.883

logistic morph_and_normal



new dataset (tanhuma, removed mishna from talmuds)
logistic normal
Train acc: 0.955
Test acc: 0.848

unified acronyms
Train acc: 0.954
Test acc: 0.825


count vectorizer
Train acc: 1.000
Test acc: 0.816
"""