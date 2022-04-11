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
from torch.utils.data import TensorDataset, DataLoader

from gensim.models import Word2Vec
import pandas as pd
from tqdm import tqdm
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

np.random.seed(42)

tqdm.pandas()

# label_names = ["MI", "HA", "JT", "BT", "AG", "TAN"]
label_names = [0, 1, 2, 3, 4, 5]
# mode = "diac"
df = pd.read_json(r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_text_morph.json")


def morph_map(x):
    if x == "":
        return "0"
    return x

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.key_to_index.keys():
        tokens.append(model.wv[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure([64, 64])
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


df.morph = df.morph.apply(morph_map)
# all_text = " ".join(df.text)
# all_morph = " ".join(df.morph)
# df_train, df_test = train_test_split(df, test_size=0.2)
#
# sentences = all_text.split(".")
# sentences = [sent + "." for sent in sentences]

# if not os.path.exists("word2vec.model"):
#     w2v_model = Word2Vec(sentences=[df.text.values.tolist()], vector_size=100, window=5, min_count=50, workers=4)
#     model2 = Word2Vec(sentences=[df.text.values.tolist()], vector_size=100, window=5, min_count=200, workers=4)
#     w2v_model.save("word2vec.model")
# else:
#     w2v_model = Word2Vec.load("word2vec.model")

# a = 5
# tsne_plot(model)

ds_path = r"C:\Users\soki\Documents\TAU\DL\proj\style\dataset_text_morph_blocks.json"
if not os.path.exists(ds_path):
    ddf = pd.DataFrame(columns=["text", "morph", "label", "book"])
    block_size = 50
    for i in range(0, len(df), block_size):
        cur_df = df.iloc[i:i+block_size]
        book = df.iloc[i:i+block_size].book.values
        if len(cur_df) < block_size:
            continue
        if cur_df.book.iloc[0] != cur_df.book.iloc[-1] or cur_df.label.iloc[0] != cur_df.label.iloc[-1]:
            continue
        text = " ".join(cur_df.text)
        morph = " ".join(cur_df.morph)
        ddf.loc[len(ddf)] = [text, morph, cur_df.label.iloc[0], cur_df.book.iloc[0]]

    ddf.to_json(ds_path)
else:
    ddf = pd.read_json(ds_path)

ddf = ddf.sample(n=1000) #todo:remove
df_train, df_test = train_test_split(ddf, test_size=0.2)

from torch import nn
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

class myLSTM(nn.Module):
    def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,output_dim, drop_prob=0.5, do_morph=False):
        super(myLSTM, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.no_layers = no_layers
        self.vocab_size = vocab_size

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.do_morph = do_morph

        #lstm
        if self.do_morph:
            self.lstm = nn.LSTM(input_size=embedding_dim + 54, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,
                                num_layers=no_layers, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        # self.sig = nn.Sigmoid()
        self.sm = nn.Softmax()

    def forward(self,x,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        logits = self.fc(out)

        return logits, hidden

    def forward2(self,x,xm,hidden):
        batch_size = x.size(0)
        # embeddings and lstm_out
        embeds = self.embedding(x)  # shape: B x S x Feature   since batch = True
        #print(embeds.shape)  #[50, 500, 1000]
        embeds = embeds + xm
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        logits = self.fc(out)

        return logits, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden


import re
def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

from collections import Counter
# def preprocess_data(x_train, y_train, x_val, y_val):
def preprocess_data(df_train, df_test, max_vocab=None):
    word_list = []

    # stop_words = set(stopwords.words('english'))
    x_train = df_train.text.values
    x_test = df_test.text.values
    stop_words = []
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    if max_vocab is not None:
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:max_vocab]
    else:
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_test:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])


    return np.array(final_list_train), df_train.label.values , np.array(final_list_test), df_test.label.values, onehot_dict



def preprocess_data2(df_train, df_test, max_vocab=None):
    word_list = []

    # stop_words = set(stopwords.words('english'))
    x_train = df_train.text.values
    x_test = df_test.text.values
    stop_words = []
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    # sorting on the basis of most common words
    if max_vocab is not None:
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:max_vocab]
    else:
        corpus_ = sorted(corpus,key=corpus.get,reverse=True)
    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}

    # tockenize
    final_list_train,final_list_test = [],[]
    final_list_train_morph,final_list_test_morph = [],[]
    for i, sent in enumerate(x_train):
        msent = df_train.morph.values[i]
        if df_train.iloc[i].text.split().__len__() == df_train.iloc[i].morph.split().__len__():
            final_list_train.append([onehot_dict[preprocess_string(word)] | (int(msent.split()[j] * (2**18)) ) for j, word in enumerate(sent.lower().split())
                                     if preprocess_string(word) in onehot_dict.keys()])
            final_list_train_morph.append([int(msent.split()[j]) for j, word in enumerate(sent.lower().split())
                                     if preprocess_string(word) in onehot_dict.keys()])
    for i, sent in enumerate(x_test):
        msent = df_test.morph.values[i]
        if df_test.iloc[i].text.split().__len__() == df_test.iloc[i].morph.split().__len__():
            final_list_test.append([onehot_dict[preprocess_string(word)] for j, word in enumerate(sent.lower().split())
                                     if preprocess_string(word) in onehot_dict.keys()])
            final_list_test_morph.append([int(msent.split()[j]) for j, word in enumerate(sent.lower().split())
                                           if preprocess_string(word) in onehot_dict.keys()])

    return np.array(final_list_train), np.array(final_list_train_morph), df_train.label.values , np.array(final_list_test), np.array(final_list_test_morph), df_test.label.values, onehot_dict


# x_train, y_train,x_test, y_test,vocab = preprocess_data(df_train, df_test)
x_train,xm_train, y_train,x_test,xm_test, y_test,vocab = preprocess_data2(df_train, df_test)
print(f'Length of vocabulary is {len(vocab)}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = lambda s: s.split() # word-based

# Text.build_vocab(train_data, max_size=max_size)
# Label.build_vocab(train_data)
# vocab_size = len(Text.vocab)
#
# train_iterator, valid_iterator, test_iterator = create_iterator(train_data, valid_data, test_data, batch_size, device)

# loss function
lr = 0.01
loss_func = nn.CrossEntropyLoss()
# lstm_model = LSTM(vocab_size, embedding_size, n_filters, filter_sizes, pool_size, hidden_size, num_classes, dropout_keep_prob)
# model = myLSTM(no_layers=1, vocab_size=len(vocab)+1, embedding_dim=100, hidden_dim=50, output_dim=len(df_train.label.unique()))
model = myLSTM(no_layers=1, vocab_size=len(vocab)+1+54, embedding_dim=100, hidden_dim=50, output_dim=len(df_train.label.unique()))
# def __init__(self,no_layers,vocab_size,hidden_dim,embedding_dim,output_dim, drop_prob=0.5):
model.to(device)

# optimization algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# function to predict accuracy
def acc(pred,label):
    # pred = torch.round(pred.squeeze())
    return torch.sum(pred == label.squeeze()).item()

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

clip = 5
epochs = 5
batch_size = 16
valid_loss_min = np.Inf
# train for some number of epochs
epoch_tr_loss,epoch_vl_loss = [],[]
epoch_tr_acc,epoch_vl_acc = [],[]

x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)
xm_train_pad = padding_(xm_train,500)
xm_test_pad = padding_(xm_test,500)

train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(xm_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(xm_test_pad), torch.from_numpy(y_test))

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

for epoch in range(epochs):
    train_losses = []
    train_acc = 0.0
    model.train()
    # initialize hidden state
    h = model.init_hidden(batch_size)
    for inputs, inputsm, labels in train_loader:

        inputs, inputsm, labels = inputs.to(device), inputsm.to(device), labels.to(device)
        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])
        h = tuple([e[:, :len(inputs)] for e in h]) #if batch size changes

        model.zero_grad()
        output,h = model.forward2(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels)
        loss.backward(retain_graph=True)
        train_losses.append(loss.item())
        # calculating accuracy
        preds = output.argmax(dim=1)
        accuracy = acc(preds,labels)
        train_acc += accuracy
        #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()



    val_h = model.init_hidden(batch_size)
    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, inputsm, labels in valid_loader:
        val_h = tuple([each.data for each in val_h])
        val_h = tuple([e[:, :len(inputs)] for e in val_h]) #if batch size changes

        # inputs, labels = inputs.to(device), labels.to(device)
        inputs, inputsm, labels = inputs.to(device), inputsm.to(device), labels.to(device)

        # output, val_h = model(inputs, val_h)
        output, val_h = model.forward2(inputs, inputsm, val_h)
        val_loss = criterion(output.squeeze(), labels)

        val_losses.append(val_loss.item())

        preds = output.argmax(dim=1)
        accuracy = acc(preds,labels)
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc/len(train_loader.dataset)
    epoch_val_acc = val_acc/len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch+1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
    if epoch_val_loss <= valid_loss_min:
        torch.save(model.state_dict(), './working/state_dict.pt')
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss))
        valid_loss_min = epoch_val_loss
    print(25*'==')



# https://galhever.medium.com/sentiment-analysis-with-pytorch-part-4-lstm-bilstm-model-84447f6c4525