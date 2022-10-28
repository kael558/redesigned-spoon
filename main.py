import os

import cohere
import pandas as pd
import sklearn.cluster
import umap

from annoy import AnnoyIndex
import numpy as np
from dotenv import load_dotenv
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def get_key():
    load_dotenv()
    return os.getenv("COHERE_KEY")


def getDataFrame(datafile: str) -> pd.DataFrame:
    return pd.read_csv(datafile, encoding="ISO-8859-1")

def getEmbeddings(co: cohere.Client, df: pd.DataFrame) -> np.array:
    embeds = co.embed(texts=list(df['Summary']), model='large', truncate='right').embeddings

    embeds = np.array(embeds)
    return embeds

def fitModel(embeddings: np.array):
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    model = model.fit(embeddings)
    return model

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

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot2DChart(df, umap_embeds):
    df_explore = pd.DataFrame(data={'text': df['Title'], 'subject': df['Subject']})
    df_explore['x'] = umap_embeds[:, 0]
    df_explore['y'] = umap_embeds[:, 1]
    mapping = {'Astrophysics':0, 'Mathematics':1, 'q-bio':2, 'Economics':3, 'Statistics':4}

    df_explore['subject'] = df_explore['subject'].map(mapping)

    # Plot
    df_explore.plot.scatter(x='x', y='y',  color='subject', colormap='viridis')
    plt.show()



def main():
    key = get_key()
    co = cohere.Client(key)

    # Get dataframe
    df = getDataFrame('data_100.csv')

    # Get vectors using coheres embeddings
    embeddings = getEmbeddings(co, df)

    # Cluster them using dendrograms & Plot them
    model = fitModel(embeddings)
    plotDendrogram(model)

    # Map each embedding to 2d
    reducer = umap.UMAP()
    umap_embeds = reducer.fit_transform(embeddings)

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
