import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

df = pd.read_csv('data/prompts_responses.csv')
df = df[["Teacher ID", "Original Prompt", "Output #1 (Andrea)"]]
df = df.rename(columns={"Output #1 (Andrea)": "Response", "Original Prompt": "Prompt"})

grouped_prompts = df.groupby("Teacher ID")["Prompt"].apply(list).tolist()
print(grouped_prompts[0])

def count_clusters(prompts, distance_threshold=0.7):
    if len(prompts) < 2:
        return len(prompts)
    
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompts)
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Count unique clusters
    num_clusters = len(set(cluster_labels))
    
    return num_clusters



for i in range(len(grouped_prompts)):
    num_clusters = count_clusters(grouped_prompts[i], 1)
    print(f"Teacher ID: {i+1}. Number of prompts: {len(grouped_prompts[i])}. Number of clusters: {num_clusters}")