from gensim.models import Word2Vec
import pandas as pd
import os
from pprint import pprint
from bidi.algorithm import get_display
import os
import gensim
import smart_open

def read_corpus(fname, tokens_only=False):
    # with smart_open.open(fname, encoding="iso-8859-1") as f:
    with smart_open.open(fname, encoding="utf8") as f:
        for i, line in enumerate(f):
            tokens = gensim.utils.simple_preprocess(line)
            if tokens_only:
                yield tokens
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])




# Set file names for train and test data
test_data_dir = os.path.join(gensim.__path__[0], 'test', 'test_data')
lee_train_file = os.path.join(test_data_dir, 'lee_background.cor')

df = pd.read_json("dataset.json")
with open(lee_train_file, "w", encoding="utf8") as f:
    f.write("\n".join(df.text.values.tolist()))

train_corpus = list(read_corpus(lee_train_file))
# test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

print(train_corpus[:2])

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save("doc2vec.model")

vector = model.infer_vector(['כדי', 'להרחיק', 'את', 'האדם', 'מן', 'העברה'])
print(vector)
# sents = df.text.values.tolist()
# common_texts = [sent.replace(",","").replace(".","").split() for sent in sents]
# common_texts = common_texts[0]
