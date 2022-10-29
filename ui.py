#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from main import *

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

df.loc[(num_nearest+1)] = ['Query', query, '']
all_embeddings = np.vstack([nn_embeddings, query_embed])

# Cluster them using dendrograms & Plot them
model = fitModel(nn_embeddings)
#model.children_

# Map the nearest embeddings to 2d
umap_embeds = getUMAPEmbeddings(all_embeddings)

# Plot points on 2d chart
fig = plot2DChart(df, umap_embeds)

st.plotly_chart(fig)

age = st.slider('Hierarchical cluster slider', min_value=0, max_value=num_nearest, step=1, value=num_nearest)
