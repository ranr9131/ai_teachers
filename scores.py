import pandas as pd
from metrics import *
import json
from joblib import Parallel, delayed

df = pd.read_csv('data/prompts_responses.csv')
df = df[["Original Prompt", "Output #3 (Andrea)"]]
df = df.rename(columns={"Output #3 (Andrea)": "Response", "Original Prompt": "Prompt"})



print(df.head())


prompt_scores = {}  # prompt text : score list
response_scores = {}    # response text : score list

def compute_prompt_scores(prompt):
    return {
        "prompt_length_score": prompt_length_score(prompt),
        "first_order_coherence_score": first_order_coherence_score(prompt),
        "second_order_coherence_score": second_order_coherence_score(prompt),
        "flesch_readability_score": flesch_readability_score(prompt),
        "readability_index_score": readability_index_score(prompt),
        "dependency_distance_score": dependency_distance_score(prompt),
        "formality_score": prompt_formality_score(prompt)
    }

def compute_response_scores(response):
    return {
        "first_order_coherence_score": first_order_coherence_score(response),
        "second_order_coherence_score": second_order_coherence_score(response),
        "flesch_readability_score": flesch_readability_score(response),
        "readability_index_score": readability_index_score(response),
        "dependency_distance_score": dependency_distance_score(response),
        "formality_score": prompt_formality_score(response),
        "depth_score": response_depth_score(response)
    }

for i in range(len(df)):
    print(i)

    # prompt = df["Prompt"][i]
    # prompt_scores[prompt] = compute_prompt_scores(prompt)

    response = df["Response"][i]
    response_scores[response] = compute_response_scores(response)



# with open('prompt_scores.json', 'w', encoding='utf-8') as f:
#     json.dump(prompt_scores, f, ensure_ascii=False, indent=4)

with open('response_scores3.json', 'w', encoding='utf-8') as f:
    json.dump(response_scores, f, ensure_ascii=False, indent=4)