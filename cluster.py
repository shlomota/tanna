import pandas as pd
import numpy as np
import random
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

BASE_PATH = r"C:\Users\soki\Documents\TAU\Thesis\Mishnah"
SEDER_LIST = ["zeraim", "moed", "nashim", "nezikin", "kodashim", "tahorot"]
# SEDER_LIST = [get_display(name) for name in SEDER_LIST]


random.seed(42)
np.random.seed(42)

do_confusion = False
# df = pd.read_json(os.path.join(BASE_PATH, r"mishna\dataset.json"))
df = pd.read_json(os.path.join(BASE_PATH, r"dataset_en.json"))


tractate_list = list(df.tractate.unique())
tractate_2_index = {v:i for i,v in enumerate(tractate_list)}

df2 = pd.DataFrame(columns=["text", "tractate", "seder"])
texts = []
for tractate in tractate_list:
    # texts += ["\n".join(df[df.tractate == tractate]["text"])]
    seder = df[df.tractate == tractate]["seder"].iloc[0]
    df2.loc[len(df2)] = ["\n".join(df[df.tractate == tractate]["text"]), tractate, seder]


# vec = CountVectorizer(max_features=100)
vec = CountVectorizer()
# vec = TfidfVectorizer(max_features=1000)
embeds = vec.fit_transform(df2.text)
embeds = embeds.todense()
embeds = embeds / np.sum(embeds, axis=1)
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(embeds)


cdict = {1: 'red', 2: 'blue', 3: 'green', 4: "black", 5: "grey", 6: "yellow"}

fig = plt.figure()
ax = fig.add_subplot()
for g in df2.seder.unique():
    ix = df2[df2.seder == g].index
    ax.scatter(tsne_results[ix, 0], tsne_results[ix, 1], c = cdict[g], label = g, s = 10)
# ax.scatter(tsne_results[:, 0], tsne_results[:, 1], )
for i in range(len(tractate_list)):
    ax.annotate(tractate_list[i], (tsne_results[i, 0], tsne_results[i, 1]), size=8)
ax.legend()
plt.show()
a=5

# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(embeds.todense())
# # dff['pca-one'] = pca_result[:,0]
# # dff['pca-two'] = pca_result[:,1]
# # dff['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     data=pca_result,
#     legend="full",
#     alpha=0.3
# )
# plt.show()
