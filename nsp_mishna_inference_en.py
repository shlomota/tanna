#TODO: replace  ג€ with " remove <b> </b> <.*>
import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sn
import os
import re
import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, BertForNextSentencePrediction
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer

from transformers import EarlyStoppingCallback

from transformers import Trainer, TrainingArguments

# BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"


random.seed(42)
np.random.seed(42)

do_confusion = False
# BASE_PATH = r"/content/drive/MyDrive/tanna"
# df = pd.read_json("dataset.json")
df = pd.read_json("dataset_en.json")
# df = pd.read_json(os.path.join(BASE_PATH, r"dataset.json"))



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

do_filter = True
if do_filter:
    df = df[df.tractate != "avot"]

do_unite = True
if do_unite:
    df["tractate"] = df.tractate.apply(unite_tractates)

pairs_df = pd.DataFrame(columns=["text1", "text2", "tractate1", "chapter1", "index1", "transition", "pred", "label"])

for i in range(len(df) - 1):
    label = 0
    if df.iloc[i]["chapter"] == df.iloc[i+1]["chapter"]:
        transition = "none"
        label = 1
    elif df.iloc[i]["tractate"] == df.iloc[i+1]["tractate"]:
        transition = "chapter"
    elif df.iloc[i]["seder"] == df.iloc[i+1]["seder"]:
        transition = "tractate"
    else:
        transition = "seder"
    pairs_df.loc[len(pairs_df)] = [df.iloc[i].text, df.iloc[i+1].text, df.iloc[i].tractate, df.iloc[i].chapter, df.iloc[i]["index"], transition, 0.0, label]


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


# BASE_PATH = r"C:\Users\soki\Documents\TAU\DL\proj"
BASE_PATH = r"/content/drive/MyDrive/tanna"

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

model.config.type_vocab_size = 2

# Create a new Embeddings layer, with 2 possible segments IDs instead of 1
model.base_model.embeddings.token_type_embeddings = nn.Embedding(2, model.config.hidden_size)

# Initialize it
model.base_model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=model.config.initializer_range)


pairs_df = pairs_df.sample(frac=0.01) #todo: remove
X_train = pairs_df[["text1", "text2"]]
y_train = pairs_df.label.values.tolist()
print("Tokenizing data")
X_train_tokenized = tokenizer(X_train.values.tolist(), padding=True, truncation=True, max_length=511)
print("Finished tokenizing data")

train_dataset = Dataset(X_train_tokenized, y_train)

# ----- 2. Fine-tune pretrained model -----#
# Define Trainer parameters
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Define Trainer
args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="steps",
    eval_steps=50,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
    save_total_limit=1,
    learning_rate=5e-5
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)



# Train pre-trained model
preds, labels, metrics = trainer.predict(test_dataset=train_dataset)
print(metrics)

preds_pos = torch.softmax(torch.tensor(preds), 1).numpy()[:, 1].tolist()
preds = [round(a, 3) for a in preds_pos]
pairs_df.preds = preds

# do_balance = True
# if do_balance:
#     df = get_balanced_dataset(df)
a = 5
pairs_df.to_csv("nsp_en_preds.csv", encoding="utf8", index=False)