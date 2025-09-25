import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np



df = pd.read_csv('data/prompts_responses.csv')
df = df[["Original Prompt", "Output #1 (Andrea)"]]
df = df.rename(columns={"Output #1 (Andrea)": "Response"})
test_resp = df["Response"][0]


# RESPONSE METRICS
def response_score(response: str):
    pass

def response_depth_score(response: str):
    sentences = nltk.sent_tokenize(response)
    sentence_count = len(sentences)
    mean_sentence_len = sum(list(map(lambda x: len(x.split()), sentences))) / sentence_count    # len: num words

    # hierarchical
    if len(sentences) > 3:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.7)
        clusters = clustering.fit_predict(embeddings)
        num_clusters = len(set(clusters))
        mean_sentences_per_cluster = sentence_count / num_clusters

    else:
        num_clusters = 0
        mean_sentences_per_cluster = 0

    def hierarchical_score(num_clusters, mean_sentences_per_cluster, sentence_count):
        cluster_score = min(num_clusters / (sentence_count * 0.3), 1.0)    # ideal num_clusters: total_sentences * 0.3, cap at 1.0
        
        balance_score = max(0, 1.0 - abs(mean_sentences_per_cluster - 3) / 3)    # ideal mean_sentences_per_cluster: 3, floor at 0 (if mean_sentences_per_cluster too high)
        
        # if too many clusters relative to sentences, penalize (clusters not developed)
        fragmentation_ratio = num_clusters / sentence_count
        if fragmentation_ratio > 0.7:  # over 70% of sentences have their own cluster
            fragmentation_penalty = (fragmentation_ratio - 0.7) * 2  # scaling the penalty: 2
            cluster_score *= (1 - fragmentation_penalty)  # Reduce cluster score
        
        final_score = (cluster_score + balance_score) / 2
        return max(0, final_score)  # Don't go negative

    hier_score = hierarchical_score(num_clusters, mean_sentences_per_cluster, sentence_count)

    # normalizing 0-1
    sentence_count_norm = min(sentence_count / 20, 1.0)  # threshold for max score: 10 sentences
    avg_length_norm = min(mean_sentence_len / 25, 1.0)   # threshold for max score: 10 words per sentence

    # final depth score, weighted between the three scores
    depth_score = (sentence_count_norm * 0.3) + (avg_length_norm * 0.2) + (hier_score * 0.5)

    return depth_score

# test_resp = "This is sentence one. Here's sentence two! And a abadbasdvbd sentence? And a fourth!"
print(response_depth_score(test_resp))