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
        

    prompt_ids_per_cluster = []
    # Print clustering results
    print(f"\nNumber of clusters: {len(set(cluster_labels))}")
    for cluster_id in sorted(set(cluster_labels)):
        temp_cluster_id = []
        cluster_prompts = [prompts[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}:")
        for prompt in cluster_prompts:
            print(f"  - {prompts.index(prompt)+1}. {prompt}")
            temp_cluster_id.append(prompts.index(prompt)+1)
            # print(f"  - {prompts.index(prompt)}. {prompt[:60]}...")

        prompt_ids_per_cluster.append(temp_cluster_id)
    
    return linkage_matrix, cluster_labels, prompt_ids_per_cluster



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



prompts_list = list(prompt_scores.keys())


linkage_matrix, labels, prompt_ids_by_cluster = cluster_and_dendrogram_embeddings(
    prompts_list, 
    distance_threshold=2.2,
    filename="embedding_dendrogram.png"
)


df = pd.read_csv('data/prompts_responses.csv')
df = df[["Teacher ID","Prompt ID","Original Prompt", "Output #1 (Andrea)"]]
df = df.rename(columns={"Output #1 (Andrea)": "Response", "Original Prompt": "Prompt"})

print(df)


df = df.copy()
df["Sequential Prompt ID"] = range(1, len(df) + 1)
teacher_prompt_groups = df.groupby("Teacher ID")["Sequential Prompt ID"].apply(list).to_dict()
prompt_ids_by_teacher = []
for teacher_id, prompt_ids in teacher_prompt_groups.items():
    print(f"Teacher {teacher_id}: {prompt_ids}")
    prompt_ids_by_teacher.append(prompt_ids)


def get_cluster_by_id(id: int):
    for i in range(len(prompt_ids_by_cluster)):
        if id in prompt_ids_by_cluster[i]:
            return i
    return -1

print("\n\nCLUSTER SEQUENCES")
for i in range(len(prompt_ids_by_teacher)):
    teacher_cluster_seq = [get_cluster_by_id(prompt_id) for prompt_id in prompt_ids_by_teacher[i]]
    print(f"Teacher {i+1}: {teacher_cluster_seq}")
