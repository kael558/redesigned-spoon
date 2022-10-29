import os

from collections import Counter
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import cohere
import pandas as pd
import sklearn.cluster
import umap
import seaborn as sns
from annoy import AnnoyIndex
import numpy as np
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go

@st.cache
def get_key():
    load_dotenv()
    return os.getenv("COHERE_KEY")

@st.cache
def getCohereClient(key):
    return cohere.Client(key)

@st.cache(allow_output_mutation=False)
def getDataFrame(datafile: str) -> pd.DataFrame:
    return pd.read_csv(datafile, encoding="ISO-8859-1")

@st.cache
def getEmbeddings(co: cohere.Client, df: pd.DataFrame) -> np.array:
    embeds = co.embed(texts=list(df['Summary']), model='large', truncate='right').embeddings

    embeds = np.array(embeds)
    return embeds

@st.cache
def fitModel(embeddings: np.array):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(embeddings)
    return model

@st.cache
def getUMAPEmbeddings(embeddings: np.array):
    # Map the nearest embeddings to 2d
    reducer = umap.UMAP()
    return reducer.fit_transform(embeddings)

def plotDendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    return linkage_matrix

@st.cache
def getData(x, y, umap_embeds, df):
    index = np.argwhere(np.all((umap_embeds-np.array([x, y]))==0, axis=1))
    return df.iloc[index[0][0]]

@st.cache
def plot2DChart(df, umap_embeds, clusters=None):
    if clusters is None:
        clusters = {}
    df_explore = pd.DataFrame(data={'title': df['Title'], 'subject': df['Subject'], 'summary': df['Summary']})
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]

    #mapping = {'Astrophysics':0, 'Mathematics':1, 'q-bio':2, 'Economics':3, 'Statistics':4, 'Query': 5}
    #df_explore['subject'] = df_explore['subject'].map(mapping)

    # Plot
    fig = px.scatter(df_explore, x='x', y='y', color='subject', hover_data=['title'])

    for cluster in clusters.values():
        high_freq_words = str(list({x: count for x, count in cluster[2].items() if count >= 3}.keys())[:10])
        fig.add_trace(go.Scatter(
            x=cluster[0],
            y=cluster[1],
            fill="toself",
            mode='lines',
            text=high_freq_words,
            opacity=0.5,
            showlegend=False

        ))

    fig.data = fig.data[::-1]

    '''fig.add_trace(go.Scatter(x=df_explore['x'],
                             y=df_explore['y'],
                             mode="markers",
                             marker_color=df_explore['subject'],
                             text=df_explore['title'],
                             showlegend=True))'''


    # Add the clusters
    ''' for cluster in clusters:
        x0, y0, x1, y1 = cluster
        fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line_color="LightSeaGreen",

            )'''

    return fig

@st.cache
def getWordFrequencies():
    stopWords = set(stopwords.words("english"))
    df = getDataFrame('data_100.csv')
    df = df.copy(deep=True)

    def is_not_stop_word(word: str) -> bool:
        if len(word) == 1:
            return False
        return word not in stopWords

    def does_not_have_special_chars(word: str) -> bool:
        return bool(re.search('^[a-z0-9]*$', word))

    def filter_stop_words(summary: str) -> list:
        all_words_lower_case = word_tokenize(summary.lower())
        words = list(filter(lambda s: is_not_stop_word(s) and does_not_have_special_chars(s), all_words_lower_case))
        return words

    def get_word_count(words: list) -> dict:
        return Counter(words)

    df['Words'] = df['Summary'].apply(filter_stop_words)
    df['Word Count'] = df['Words'].apply(get_word_count)

    x = df['Word Count'].values

    dictionary = {}
    i = 0
    for k in x:
        dictionary[i] = k
        i+=1

    return dictionary

@st.cache
def saveBuild(embeds: np.array, indexfile: str):
    search_index = AnnoyIndex(embeds.shape[1], 'angular')

    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])

    search_index.build(10)
    search_index.save(indexfile)

@st.cache
def get_query_embed(co: cohere.Client, query: str):
    query_embed = co.embed(texts=[query],
                           model='large',
                           truncate='right').embeddings

    return np.array(query_embed)

@st.cache
def get_query_nn(indexfile: str, query_embed: np.array, neighbours=100):
    search_index = AnnoyIndex(4096, 'angular')
    search_index.load(indexfile)

    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0], neighbours,
                                                      include_distances=True)

    return similar_item_ids


def main():
    np.random.seed(42)
    key = get_key()
    co = cohere.Client(key)

    # Get dataframe
    df = getDataFrame('data_100.csv')

    # Get vectors using coheres embeddings
    embeddings = getEmbeddings(co, df)

    # Save embeddings
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
    model = fitModel(embeddings)
    linkages = plotDendrogram(model)

    # Map the nearest embeddings to 2d
    reducer = umap.UMAP()
    umap_embeds = reducer.fit_transform(all_embeddings)

    # Plot points on 2d chart
    plot2DChart(df, umap_embeds)


if __name__ == '__main__':
    main()
    '''
    1. Get vectors using coheres embeddings
    2. Cluster them using dendrograms (maybe reduce vector size using PCA/uMap) - play around with hyperparameter dimension
    3. Get the frequency analysis of documents in each cluster
    '''

    '''
    The user can then:
    1. Query -   
        - use nn to get nearest 100
        - convert to 2d
        - dendrogram for nearest 100
        - plot query as well 
    2. Concepts
        - pop-up of keywords in cluster
        - adjust level of clustering using a slider
    '''
