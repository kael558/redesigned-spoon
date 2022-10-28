import os

import cohere
import pandas as pd
from annoy import AnnoyIndex
import numpy as np
from dotenv import load_dotenv

def get_key():
    load_dotenv()
    return os.getenv("COHERE_KEY")

key = get_key()

def buildIndex():
    df = pd.read_csv('data.csv', encoding="ISO-8859-1")

    embeds = co.embed(texts=list(df['Summary']), model = 'large', truncate='right').embeddings

    embeds = np.array(embeds)

    search_index = AnnoyIndex(embeds.shape[1], 'angular')
    print(embeds.shape[1])

    for i in range(len(embeds)):
        search_index.add_item(i, embeds[i])

    search_index.build(10)
    search_index.save('test.ann')

def getClosestNeighbours():
    df = pd.read_csv('data.csv', encoding="ISO-8859-1")

    search_index = AnnoyIndex(4096, 'angular')
    search_index.load('test.ann')

    query = 'I want a paper on astro physics'

    query_embed = co.embed(texts=[query],
                        model='large',
                        truncate='right').embeddings

    # Retrieve the nearest neighbors
    similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                        include_distances=True)
    # Format the results
    print(similar_item_ids)
    results = pd.DataFrame(data={'title': df.iloc[similar_item_ids[0]]['Title'],
                                 'subject': df.iloc[similar_item_ids[0]]['Subject'],
                                  'summary': df.iloc[similar_item_ids[0]]['Summary'],
                                 'distance': similar_item_ids[1]})

    print(f"Query:'{query}'\nNearest neighbors:")
    print(results)


if __name__ == '__main__':

    co = cohere.Client(key)
    buildIndex()
    getClosestNeighbours()
