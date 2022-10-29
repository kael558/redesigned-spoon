#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from main import *
import copy

st.title('Arvix Semantic Paper Searcher')

query = st.text_input('Please input your query here: ', 'Celestial bodies and physics')

co = getCohereClient(get_key())

# Get dataframe
df = getDataFrame('data_100.csv')

# Get vectors using coheres embeddings
embeddings = getEmbeddings(co, df)

# Save embeddings as Annoy
indexfile = 'index.ann'
saveBuild(embeddings, indexfile)

# Get query embeddings and append to embeddings
query_embed = get_query_embed(co, query)

# Get nearest points
num_nearest = 100
nearest_ids = get_query_nn(indexfile, query_embed, num_nearest)
df = df.loc[nearest_ids[0]]
nn_embeddings = embeddings[nearest_ids[0]]

df.loc[(num_nearest + 1)] = ['Query', query, '']
all_embeddings = np.vstack([nn_embeddings, query_embed])

# Cluster them using dendrograms & Plot them
model = fitModel(nn_embeddings)

#linkages = plotDendrogram(model)

# Map the nearest embeddings to 2d
umap_embeds = getUMAPEmbeddings(all_embeddings)

# Get the word frequencies
word_frequencies = getWordFrequencies()

# level 0 = show each doc as own cluster, level n = 1 cluster
def get_clusters(level):
    word_frequencies_combined = copy.deepcopy(word_frequencies)
    cluster_combine_order = copy.deepcopy(model.children_)

    cluster_mappings = dict()
    for cluster in model.labels_:
        cluster_mappings[cluster] = [cluster]

    n = len(cluster_mappings)
    for i in range(level):
        indicies_of_merged_clusters = cluster_combine_order[0]
        cluster_combine_order = np.delete(cluster_combine_order, 0, axis=0)
        cluster_mappings[n] = cluster_mappings.pop(indicies_of_merged_clusters[0]) \
                              + cluster_mappings.pop(indicies_of_merged_clusters[1])

        word_frequencies_combined[n] = word_frequencies_combined.pop(indicies_of_merged_clusters[0]) \
                                       + word_frequencies_combined.pop(indicies_of_merged_clusters[1])

        n += 1

    def convex_hull(points):
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        points = sorted(set(points))

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(points) <= 1:
            return points

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list.
        return lower[:-1] + upper[:-1]

    # Calculate boundaries
    for k in cluster_mappings.keys():
        cluster_mappings[k] = map(lambda i: (umap_embeds[i][0], umap_embeds[i][1]),cluster_mappings[k])
        cluster_mappings[k] = convex_hull(cluster_mappings[k])
        x_list, y_list = [], []
        for x, y in cluster_mappings[k]:
            x_list.append(x)
            y_list.append(y)
        cluster_mappings[k] = (x_list, y_list, word_frequencies_combined[k])

    return cluster_mappings

placeholder=st.empty()
level = st.slider('Hierarchical cluster slider', min_value=0, max_value=num_nearest, step=1, value=num_nearest)

clusters = get_clusters(level-1)

with placeholder.container():
    # Plot points on 2d chart
    fig = plot2DChart(df, umap_embeds, clusters)
    st.plotly_chart(fig)







