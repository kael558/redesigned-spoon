## About The Project
The Semantic Search Tool maps research papers into points on a 2-d chart. It clusters them according to the level requested by the user. Queries are also mapped to the chart to identify the cluster that the query belongs to. 

The points are clickable so the user can see more information in the sidebar. Also, each step of the slider will show a new cluster added to the selected paper's cluster. This allows the user to see new and relevant clusters to their selected paper. 

## Use cases
 - The user may enter a query and find relevant papers
 - The user may select a paper and find relevant papers
 - The user may observe the overall clustering of all of the papers at different levels
 - The user may identify the keywords of each cluster
 - The user may see papers from other fields that are relevant to their selected paper

## Implementation Flow
 - Cohere’s transformer’s to embed the summary of each paper into a n-dimensional vector.
 - Use UMap to reduce the vectors to 2-d
 - Generate hierarchical clustering of points in 2D space using agglomerative clustering
 - Intersect the words of each paper in a cluster using frequency analysis
