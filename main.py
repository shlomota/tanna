from gensim.test.utils import common_texts
from bidi.algorithm import get_display
from gensim.models import Word2Vec
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0,100), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.most_similar(word, topn=100)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(get_display(label), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


def display_words_tsnescatterplot(model, words):

    arr = np.array([model.wv[word] for word in words])
    # # add the vector for each of the closest words to the array
    # arr = np.append(arr, np.array([model[word]]), axis=0)
    # for wrd_score in close_words:
    #     wrd_vector = model[wrd_score[0]]
    #     word_labels.append(wrd_score[0])
    #     arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    tsne = PCA(n_components=2)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(words, x_coords, y_coords):
        plt.annotate(get_display(label), xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()

if not os.path.exists("word2vec.model"):
    with open(r"C:\Users\soki\Documents\TAU\DL\proj\mishna\all2.txt", "r", encoding="utf-8") as f:
        data = f.readlines()
    tokenized_data = [line.split() for line in data]

    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=5, min_count=1, workers=1)
    model.save("word2vec.model")

else:
    model = Word2Vec.load("word2vec.model")

a=5
words = ["יהודה", "שמעון", "הלל", "שמאי", "אליעזר", "אלעזר", "יהושע", "טרפון", "ישמעאל", "גמליאל", "זומא", "עזאי", "רבי", "עקיבא", "יוחנן"]
words += ["אוכל", "שבת"]
display_words_tsnescatterplot(model, words)
# display_closestwords_tsnescatterplot(model, "יהודה")