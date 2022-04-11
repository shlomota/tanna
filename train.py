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

tractate_list = ["זרעים", "מועד", "נשים", "נזיקין", "קודשים", "טהרות", "אחר"]
tractate_list = ["זרעים", "מועד", "נשים", "נזיקין", "קודשים", "טהרות"]
tractate_list = [get_display(name) for name in tractate_list]

random.seed(42)
np.random.seed(42)

do_confusion = False
# df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\dataset.json")
df = pd.read_json(r"C:\Users\soki\Documents\TAU\Thesis\Mishna\dataset_en.json")
df = df.sample(frac=1)
accs = []
for test_size in np.arange(0.1,1.0,0.1):
    train_df, test_df = train_test_split(df, test_size=test_size)
    # train_df=df

    X_train, y_train = train_df.text.to_numpy(), train_df.seder.to_numpy()
    X_test, y_test = test_df.text.to_numpy(), test_df.seder.to_numpy()

    # vec = TfidfVectorizer()
    vec = CountVectorizer()
    # vec = TfidfVectorizer(ngram_range=(1,2))
    vec.fit(df.text)

    # model = LogisticRegression(fit_intercept=False)
    model = LogisticRegression()
    # model = Pipeline([("feature_selection", SelectFromModel(estimator=model,threshold=1)), ("model", model)])
    # model = Pipeline([vec, model])
    # model = RandomForestClassifier()
    # model = XGBClassifier()
    model.fit(vec.transform(X_train), y_train)

    try:
        weights = eli5.explain_weights_df(model, vec=vec)
    except:
        weights = None

    train_preds = model.predict(vec.transform(X_train))
    print("Train acc: %.3f" % accuracy_score(y_train, train_preds))

    preds = model.predict(vec.transform(X_test))
    print("Test acc: %.3f" % accuracy_score(y_test, preds))

    joblib.dump(model, "log_mishna_clf.pkl")
    joblib.dump(vec, "output/mishna_vec.pkl")
    # weights.to_csv("bigram_logistic_weights.csv", encoding="utf8")

    accs += [accuracy_score(y_test, preds)]
    # plot confusion matrices
    #TODO: parameter order? train?
    if do_confusion:
        cm = confusion_matrix(train_preds, y_train, normalize="true")
        cm2 = cm.copy()
        # sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".2g", xticklabels=range(1, len(cm2)+1), yticklabels=range(1, len(cm2)+1))
        sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt=".2g", xticklabels=tractate_list, yticklabels=tractate_list)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix normalized by true label")
        plt.show()
        # plt.savefig("cm_norm.png")

        cm = confusion_matrix(train_preds, y_train)
        cm2 = cm.copy()
        sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="g", xticklabels=tractate_list, yticklabels=tractate_list)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix absolute")
        # plt.show()
        # plt.savefig("cm_abs.png")


        cm2 = cm.copy()
        for i in range(len(cm2)):
            cm2[i,i]=0
        sn.heatmap(cm2, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Reds, fmt="g", xticklabels=tractate_list, yticklabels=tractate_list)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix absolute errors only")
        # plt.show()
        # plt.savefig("cm_abs_errors.png")

        # plot_confusion_matrix(model, vec.transform(X_train), y_train, cmap=plt.cm.Reds)
        # plt.show()
        # plt.savefig("cm.png")
        a=5


plt.plot(np.arange(0.1,1.0,0.1), accs, marker="o")
plt.ylabel("Test Accuracy")
plt.xlabel("Test dataset ratio")
plt.title("Accuracy as a function of test size")
plt.show()

"""
Logistic:
Unigram:
Train acc: 0.947
Test acc: 0.708
Count unigram:
Train acc: 1.000
Test acc: 0.721
Bigram:
Train acc: 0.966
Test acc: 0.710
Trigram:
Train acc: 0.962
Test acc: 0.671


K=100
Train acc: 0.582
Test acc: 0.513
k=1000
Train acc: 0.908
Test acc: 0.638

No intercept:
Train acc: 1.000
Test acc: 0.714

Random forrest:
Train acc: 1.000
Test acc: 0.658

Bigram:
Train acc: 1.000
Test acc: 0.659

XGB:
Train acc: 0.993
Test acc: 0.681

Bigram:
Train acc: 0.997
Test acc: 0.675
"""