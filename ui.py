#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from main import *
import copy

st.title('Paper Searcher')

text_input = st.text_input('Please input your query here: ', '''Celestial mechanics''')

co = getCohereClient(get_key())

# Get dataframe
df = getDataFrame('data_100.csv')

# Get vectors using coheres embeddings
embeddings = getEmbeddings(co, df)

# Save embeddings as Annoy
indexfile = 'index.ann'
#saveBuild(embeddings, indexfile)

# Get query embeddings and append to embeddings
query = 'Celestial bodies and physics'
query_embed = get_query_embed(co, query)

# Get nearest points
num_nearest = 100
nearest_ids = get_query_nn(indexfile, query_embed, num_nearest)
df = df.loc[nearest_ids[0]]
nn_embeddings = embeddings[nearest_ids[0]]

df.loc[(num_nearest+1)] = ['Query', query, '']
all_embeddings = np.vstack([nn_embeddings, query_embed])

# Cluster them using dendrograms & Plot them
model = fitModel(nn_embeddings)


# level 0 = show each doc as own cluster, level n = 1 cluster
def get_clusters(cluster_dict, cluster_combine_order, level=count):
    clusters = cluster_dict.copy()
    n = len(clusters)
    for i in range(level):
        values = cluster_combine_order[0]
        cluster_combine_order = np.delete(cluster_combine_order, 0, axis=0)
        clusters[n] = clusters[values[0]] + clusters[values[1]]
        clusters.pop(values[0])
        clusters.pop(values[1])
        n+=1
    return clusters

cluster_combine_order = copy.deepcopy(model.children_)

clusters = dict()
for cluster in model.labels_:
    clusters[cluster] = [cluster]

n = cluster_combine_order

print(get_clusters(clusters, cluster_combine_order, 8))




linkages = plotDendrogram(model)

# Map the nearest embeddings to 2d
umap_embeds = getUMAPEmbeddings(all_embeddings)

# Plot points on 2d chart
fig = plot2DChart(df, umap_embeds)

st.plotly_chart(fig)
