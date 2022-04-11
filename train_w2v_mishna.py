from gensim.models import Word2Vec
import pandas as pd
import os
from pprint import pprint
from bidi.algorithm import get_display
from tqdm import tqdm
from adjustText import adjust_text
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

df = pd.read_json("dataset.json")
sents = df.text.values.tolist()

do_names = True
if do_names:
    with open("names.txt", "r") as f:#, encoding="utf8") as f:
        names = f.readlines()
    names = sorted(names, key=len)[::-1]
    names = [name.strip() for name in names]
    full_text = "\n".join(sents)
    for name in tqdm(names):
        for letter in "משהוכלב":
            full_text = full_text.replace(letter + name, letter + " " + name) #separate letters from name
        full_text = full_text.replace(name, name.replace(" ", "_"))
    sents = full_text.split("\n")

common_texts = [sent.replace(",","").replace(".","").split() for sent in sents]
# common_texts = common_texts[0]

def look(word):
    pprint(model.wv.most_similar(word, topn=10))

if True or not os.path.exists("word2vec.model"):
    # model = Word2Vec(sentences=common_texts, vector_size=100, window=5, max_final_vocab=300, workers=4)
    model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
    model.save("word2vec.model")
else:
    model = Word2Vec.load("word2vec.model")

try:
    check_names = ["בית_הלל", "בית_שמאי"]
    check_names = ["רבי_אליעזר", "רבן_גמליאל", "רבי_יהושע"]
    check_names = ["הלל", "שמאי"]
    for name in check_names:
        print(name)
        # print(model.wv.most_similar("רבי_עקיבא"))
        result = model.wv.most_similar(name)
        words = [a[0] for a in result]
        print(words)
except:
    pass


vocab = list(model.wv.key_to_index)
if do_names:
    underscore_names = [name.replace(" ", "_") for name in names]
    vocab = list(filter(lambda x: x in underscore_names, vocab))
    # vocab = vocab[:50]

def get_count(term):
    res = re.findall("[^_]" + term + "[^_]", full_text)
    return len(res)




do_common = False
if do_common:
    thresh = 10
    vocab2 = list(filter(lambda x: get_count(x) > thresh, vocab))
else:
    vocab2 = vocab



X = model.wv[vocab]

tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)

df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
df = df[df.index.isin(vocab2)]


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'], marker=".")

texts = []
for word, pos in df.iterrows():
    texts.append(plt.text(pos[0], pos[1], get_display(word), size=6))
    # ax.annotate(get_display(word), pos, size=6)
plt.title("TSNE of mishna word2vec")
adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=0.5))
plt.show()