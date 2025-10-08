import pandas as pd
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from transformers import XLMRobertaTokenizerFast, XLMRobertaForSequenceClassification
import textdescriptives as td




df = pd.read_csv('data/prompts_responses.csv')
df = df[["Original Prompt", "Output #1 (Andrea)"]]
df = df.rename(columns={"Output #1 (Andrea)": "Response"})



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


# PROMPT METRICS
def prompt_formality_score(prompt: str):
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('s-nlp/xlmr_formality_classifier')
    model = XLMRobertaForSequenceClassification.from_pretrained('s-nlp/xlmr_formality_classifier')

    encoding = tokenizer(
        prompt,
        add_special_tokens=True,
        return_token_type_ids=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    output = model(**encoding)
    
    # Get probabilities with softmax
    probabilities = output.logits.softmax(dim=1)[0]  # [0] gets first (only) result
    
    # probabilities[0] = formal probability
    # probabilities[1] = informal probability    
    return probabilities[0].item()

def prompt_politeness_score(prompt: str):
    pass

def prompt_length_score(prompt: str):
    return len(prompt.split())      # only amount of words for now

def first_order_coherence_score(text: str):
    # specify spaCy model and which metrics to extract
    test = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["coherence"])

    return test['first_order_coherence'][0]

def second_order_coherence_score(text: str):
    # specify spaCy model and which metrics to extract
    test = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["coherence"])

    return test['second_order_coherence'][0]

def flesch_readability_score(text: str):
    test = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["readability"])

    return test['flesch_reading_ease'][0]

def readability_index_score(text: str):
    test = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["readability"])

    return test['automated_readability_index'][0]

def dependency_distance_score(text: str):
    test = td.extract_metrics(text=text, spacy_model="en_core_web_lg", metrics=["dependency_distance"])
    return test['dependency_distance_mean'][0]



# test = td.extract_metrics(text=df["Response"][0], spacy_model="en_core_web_lg", metrics=["dependency_distance", "coherence"])
# print(test.columns)
# print(test['dependency_distance_mean'])
# print(test['automated_readability_index'])

# print(coherence_score(df["Response"][0]))


if __name__ == "__main__":
    sentence1 = df["Response"][0]
    sentence2 = "fuck you hoe"
    print(response_depth_score(sentence1))
    print(sentence2)
    print(response_depth_score(sentence2))


    print(prompt_formality_score(sentence2))