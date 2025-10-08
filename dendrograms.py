import json
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np


with open('data/prompt_scores.json', 'r', encoding='utf8') as f:
    prompt_scores = json.load(f)

with open('data/response_scores.json', 'r', encoding='utf8') as f:
    response_scores = json.load(f)


def return_prompt_score_vector(prompt_scores, prompt: str):
    scores_dict = prompt_scores[prompt]
    return [scores_dict["prompt_length_score"], scores_dict["flesch_readability_score"], scores_dict["readability_index_score"], scores_dict["dependency_distance_score"], scores_dict["formality_score"]]

def return_response_score_vector(response_scores, response: str):
    scores_dict = response_scores[response]
    return [scores_dict["first_order_coherence_score"], scores_dict["second_order_coherence_score"], scores_dict["flesch_readability_score"], scores_dict["readability_index_score"], scores_dict["dependency_distance_score"], scores_dict["formality_score"], scores_dict["depth_score"]]




def cluster_and_dendrogram_embeddings(prompts, distance_threshold=0.7, filename = "dendrogram.png"):
    """
    Performs agglomerative clustering and creates dendrogram.
    
    Parameters:
    - prompts: List of prompt texts
    - distance_threshold: Threshold for cluster merging
    """
    
    # Get embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(prompts)
    
    # Create linkage matrix
    linkage_matrix = linkage(embeddings, method='ward')
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Create dendrogram with color threshold
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=[f"Prompt {i+1}" for i in range(len(prompts))],
        leaf_rotation=90,
        leaf_font_size=5,
        color_threshold=distance_threshold,  # Colors different clusters
        above_threshold_color='gray'
    )
    
    plt.title(f"Prompt Clustering (Distance Threshold: {distance_threshold})", fontsize=14)
    plt.xlabel("Prompt Index", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.axhline(y=distance_threshold, c='red', linestyle='--', label=f'Threshold: {distance_threshold}')
    plt.legend()
    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Dendrogram saved to {filename}")
        
    # Print clustering results
    print(f"\nNumber of clusters: {len(set(cluster_labels))}")
    for cluster_id in sorted(set(cluster_labels)):
        cluster_prompts = [prompts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}:")
        for prompt in cluster_prompts:
            print(f"  - {prompts.index(prompt)}. {prompt[:60]}...")
    
    return linkage_matrix, cluster_labels



def cluster_and_dendrogram_score_vector(score_vectors, labels=None, distance_threshold=0.7, filename = "dendrogram.png"):
    """
    Performs agglomerative clustering and creates dendrogram from embeddings.
    
    Parameters:
    - embeddings: List of lists or numpy array of embedding vectors
    - labels: Optional list of labels for each embedding (e.g., prompt names)
    - distance_threshold: Threshold for cluster merging
    """
    
    # Convert to numpy array if needed
    score_vectors = np.array(score_vectors)
    
    # Generate default labels if not provided
    if labels is None:
        labels = [f"Item {i+1}" for i in range(len(score_vectors))]
    
    # Create linkage matrix
    linkage_matrix = linkage(score_vectors, method='ward')
    
    # Perform clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold,
        linkage='ward'
    )
    cluster_labels = clustering.fit_predict(score_vectors)
    
    # Create dendrogram with color threshold
    plt.figure(figsize=(14, 8))
    dendrogram(
        linkage_matrix,
        labels=[f"Item {i+1}" for i in range(len(score_vectors))],
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=distance_threshold,
        above_threshold_color='gray'
    )
    
    plt.title(f"Clustering Dendrogram (Distance Threshold: {distance_threshold})", fontsize=14)
    plt.xlabel("Item Index", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.axhline(y=distance_threshold, c='red', linestyle='--', label=f'Threshold: {distance_threshold}')
    plt.legend()
    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Dendrogram saved to {filename}")
        
    # Print clustering results
    print(f"\nNumber of clusters: {len(set(cluster_labels))}")
    for cluster_id in sorted(set(cluster_labels)):
        cluster_items = [labels[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}:")
        for item in cluster_items:
            print(f"  - {labels.index(item)}. {item}")
    
    return linkage_matrix, cluster_labels


# prompts = [
#     "Help me create a lesson plan for fractions.",
#     "I need help teaching fractions to 5th graders.",
#     "Can you explain photosynthesis?",
#     "How do plants photosynthesize?",
#     "Give me math homework problems.",
#     "Create algebra practice problems.",
# ]

prompts_list = list(prompt_scores.keys())

score_vectors = [return_prompt_score_vector(prompt_scores, prompt) for prompt in prompts_list]

linkage_matrix, labels = cluster_and_dendrogram_score_vector(
    score_vectors, 
    labels=prompts_list,
    distance_threshold=2.0,
    filename="score_vector_dendrogram.png"
)

linkage_matrix, labels = cluster_and_dendrogram_embeddings(
    prompts_list, 
    distance_threshold=2.0,
    filename="embedding_dendrogram.png"
)


# print(linkage_matrix)
# print(labels)





# print(return_response_score_vector(response_scores, 'Here’s the illustration of a teacher demonstrating the marshmallow launch experiment in a high school science lab. This image shows the teacher explaining how to use a homemade catapult to explore projectile motion concepts, set against a classroom environment equipped with typical lab materials. This visual could be effectively used in your presentation to demonstrate the instructional part of the lab activity.'))
