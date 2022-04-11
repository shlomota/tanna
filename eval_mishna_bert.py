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
import random
import joblib
from sklearn.feature_selection import SelectKBest, SelectFromModel, chi2
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import KFold
import os
import re
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import pipeline
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import EarlyStoppingCallback

from transformers import Trainer, TrainingArguments

# training_args = TrainingArguments(
#     output_dir="./alephbert/alephbert",
#     overwrite_output_dir=True,
#     num_train_epochs=40,
#     per_gpu_train_batch_size=16,
#     save_steps=10_000,
#     prediction_loss_only=True,
# )

BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
BASE_PATH = r"/content/drive/MyDrive/tanna"
SEDER_LIST = ["זרעים", "מועד", "נשים", "נזיקין", "קודשים", "טהרות"]

random.seed(42)
np.random.seed(42)

# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
df = pd.read_json(os.path.join(BASE_PATH, r"dataset.json"))
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



do_balance = True
if do_balance:
    df = get_balanced_dataset(df)

df.seder = df.seder - 1

# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


# model_name = "bert-base-uncased"
# model_name = r"C:\Users\soki\PycharmProjects\QFIB\alephbert\alephbert - Copy"
model_name = r"onlplab/alephbert-base"
model_name = os.path.join(BASE_PATH, r"alephbert")
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)

X = list(df["text"])
y = list(df["seder"])
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print("Tokenizing data")
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)
print("Finished tokenizing data")

train_dataset = Dataset(X_train_tokenized, y_train)
val_dataset = Dataset(X_val_tokenized, y_val)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="macro")
    precision = precision_score(y_true=labels, y_pred=pred, average="macro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="macro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    save_total_limit=1
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)



# Train pre-trained model
trainer.train()

