import os

import cohere
import pandas as pd
from annoy import AnnoyIndex
from dotenv import load_dotenv

def get_key():
    load_dotenv()
    return os.getenv("COOKIE")

key = get_key()

df = pd.read_csv('data.csv')

co = cohere.Client(key)

embeds = co.embed(texts=list(df['text'], model = 'large', truncate='right')).embeddings

search_index = AnnoyIndex(embeds.shape[1], 'angular')

for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])

search_index.build(10)

query = 'I want a paper on astro physics'

query_embed = co.embed(texts=[query],
                       model='large',
                       truncate='right').embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                include_distances=True)
# Format the results
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'],
                             'distance': similar_item_ids[1]})


print(f"Query:'{query}'\nNearest neighbors:")
print(results)
