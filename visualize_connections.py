import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv("output/mishna_tosefta_links.csv")
a=5

def get_node_name(name):
    # result = re.search("(?: )(.*)(?: \d)", name).group(1)
    result = re.search("(.*)(?: \d)", name).group(1)
    return result

df["node1"] = df["Citation 1"].apply(get_node_name)
df["node2"] = df["Citation 2"].apply(get_node_name)

df[["node1", "node2"]].to_csv("mt_links.csv", index=False)
df = df[df["node1"].isin(np.random.choice(df.node1.unique(), size=1))]

from collections import Counter
a = df[["node1", "node2"]].values.tolist()
a = [tuple(b) for b in a]
c = Counter(a)

df["weight"] = df.apply(lambda x: c[(x.node1, x.node2)], axis=0
                        )

plt.figure(figsize=(16,10))

G = nx.Graph()
G.add_nodes_from(df.node1.unique(), bipartite=0)
G.add_nodes_from(df.node2.unique(), bipartite=1)
G.add_edges_from(df[["node1", "node2"]].values.tolist())

for u, v, d in G.edges(data=True):
    d['weight'] = c[u,v]
# G.add_weighted_edges_from(df[["node1", "node2", "weight"]].values.tolist())

X, Y = df["node1"], df["node2"]
pos = dict()
pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
nx.draw(G, with_labels=True, font_size=12, pos=pos)

edge_labels = nx.get_edge_attributes(G,'weight')

# Draw the edge labels
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)


# G = nx.MultiDiGraph()
# G = nx.Graph()
# G.add_nodes_from(df.node1.unique(), bipartite=0)
# G.add_nodes_from(df.node2.unique(), bipartite=1)
# G.add_edges_from(df[["node1", "node2"]].values.tolist())
# # G.add_weighted_edges_from(df[["node1", "node2", "weight"]].values.tolist())
#
# for u, v, d in G.edges(data=True):
#     d['weight'] = c[u,v]
#
# # for u,v,d in G.edges(data=True):
# #     print(u,v,d)
#
# X, Y = df["node1"], df["node2"]
# pos = dict()
# pos.update( (n, (1, i)) for i, n in enumerate(X) ) # put nodes from X at x=1
# pos.update( (n, (2, i)) for i, n in enumerate(Y) ) # put nodes from Y at x=2
# nx.draw(G, with_labels=True, font_weight='bold', pos=pos)
#
# edge_labels = nx.get_edge_attributes(G,'weight')
#
# # Draw the edge labels
# # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
# nx.draw_networkx(G, pos, width=[d["weight"] for _, _, d in G.edges(data=True)])


# nx.draw_networkx(
#     G,
#     pos = nx.drawing.layout.bipartite_layout(B, B_first_partition_nodes),
#     width = edge_widths*5) # Or whatever other display options you like

plt.show()