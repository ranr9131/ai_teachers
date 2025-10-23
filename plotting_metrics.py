import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



with open('data/connectedness_scores.json', 'r') as f:
    connectedness_scores = json.load(f)

scores = list(connectedness_scores.values())

num_sentences = [score[0] for score in scores]
num_clusters = [score[1] for score in scores]
connectedness = [score[2] for score in scores]

print(connectedness)


with open('data/prompt_scores.json', 'r', encoding='utf-8') as f:
    prompt_scores = json.load(f)

with open('data/response_scores.json', 'r', encoding='utf-8') as f:
    response_scores = json.load(f)


#prompt scores
prompt_length_scores = [score_dict["prompt_length_score"] for score_dict in list(prompt_scores.values())]
flesch_readability_scores = [score_dict["flesch_readability_score"] for score_dict in list(prompt_scores.values())]
readability_index_scores = [score_dict["readability_index_score"] for score_dict in list(prompt_scores.values())]
dependency_distance_scores = [score_dict["dependency_distance_score"] for score_dict in list(prompt_scores.values())]
formality_scores = [score_dict["formality_score"] for score_dict in list(prompt_scores.values())]

# depth scores
depth_scores = [score_dict["depth_score"] for score_dict in list(response_scores.values())]
first_order_coherence_score = [score_dict["first_order_coherence_score"] for score_dict in list(response_scores.values())]
second_order_coherence_score = [score_dict["second_order_coherence_score"] for score_dict in list(response_scores.values())]


df = pd.read_csv('data/prompts_responses.csv')
teacher_ids = df["Teacher ID"].tolist()
print(teacher_ids)

def get_indices_of_id(teacher_ids, id):
    return [i for i, t_id in enumerate(teacher_ids) if t_id == id]


teacher_avg_depth_scores = [sum([depth_scores[ind] for ind in get_indices_of_id(teacher_ids, i+1)])/len(get_indices_of_id(teacher_ids, i+1)) for i in range(24)]
teacher_avg_first_order_coherence_scores = [sum([first_order_coherence_score[ind] for ind in get_indices_of_id(teacher_ids, i+1)])/len(get_indices_of_id(teacher_ids, i+1)) for i in range(24)]
teacher_avg_second_order_coherence_scores = [sum([second_order_coherence_score[ind] for ind in get_indices_of_id(teacher_ids, i+1)])/len(get_indices_of_id(teacher_ids, i+1)) for i in range(24)]



# Scatter plot: Connectedness vs Teacher Avg First Order Coherence Scores
connectedness_arr = np.array(connectedness)
teacher_avg_first_order_coherence_scores_arr = np.array(teacher_avg_first_order_coherence_scores)
valid_mask1 = ~(np.isnan(connectedness_arr) | np.isnan(teacher_avg_first_order_coherence_scores_arr))
x1 = connectedness_arr[valid_mask1]
y1 = teacher_avg_first_order_coherence_scores_arr[valid_mask1]

plt.figure()
plt.scatter(x1, y1, alpha=0.7)
if len(x1) > 1:
    m1, b1 = np.polyfit(x1, y1, 1)
    x_line1 = np.linspace(min(x1), max(x1), 100)
    y_line1 = m1 * x_line1 + b1
    plt.plot(x_line1, y_line1, color='red', linewidth=2, label='Line of Best Fit')
plt.xlabel('Connectedness')
plt.ylabel('Teacher Avg First Order Coherence Score')
plt.title('Connectedness vs Teacher Avg First Order Coherence Score')
plt.legend()
plt.tight_layout()
plt.savefig('metric_plots/connectedness_vs_teacher_avg_first_order_coherence_score.png')
plt.close()

# Scatter plot: Connectedness vs Teacher Avg Second Order Coherence Scores
teacher_avg_second_order_coherence_scores_arr = np.array(teacher_avg_second_order_coherence_scores)
valid_mask2 = ~(np.isnan(connectedness_arr) | np.isnan(teacher_avg_second_order_coherence_scores_arr))
x2 = connectedness_arr[valid_mask2]
y2 = teacher_avg_second_order_coherence_scores_arr[valid_mask2]

plt.figure()
plt.scatter(x2, y2, alpha=0.7)
if len(x2) > 1:
    m2, b2 = np.polyfit(x2, y2, 1)
    x_line2 = np.linspace(min(x2), max(x2), 100)
    y_line2 = m2 * x_line2 + b2
    plt.plot(x_line2, y_line2, color='red', linewidth=2, label='Line of Best Fit')
plt.xlabel('Connectedness')
plt.ylabel('Teacher Avg Second Order Coherence Score')
plt.title('Connectedness vs Teacher Avg Second Order Coherence Score')
plt.legend()
plt.tight_layout()
plt.savefig('metric_plots/connectedness_vs_teacher_avg_second_order_coherence_score.png')
plt.close()



# score_lists = [
#     ("prompt_length_score", prompt_length_scores),
#     ("flesch_readability_score", flesch_readability_scores),
#     ("readability_index_score", readability_index_scores),
#     ("dependency_distance_score", dependency_distance_scores),
#     ("formality_score", formality_scores)
# ]

# # Plot vs depth_score
# for score_name, x_scores in score_lists:
#     plt.figure()
#     plt.scatter(x_scores, depth_scores, alpha=0.7)
#     m, b = np.polyfit(x_scores, depth_scores, 1)
#     plt.plot(x_scores, np.array(x_scores)*m + b, color='red')
#     plt.xlabel(score_name.replace('_', ' ').title())
#     plt.ylabel('Depth Score')
#     plt.title(f'{score_name.replace("_", " ").title()} vs Depth Score')
#     plt.tight_layout()
#     plt.savefig(f'metric_plots/{score_name}_vs_depth_score.png')
#     plt.close()

# Plot vs first_order_coherence_score
# for score_name, x_scores in score_lists:
#     plt.figure()
    
#     # Filter out NaN values
#     x_array = np.array(x_scores)
#     y_array = np.array(first_order_coherence_score)
    
#     # Create mask for valid (non-NaN) values
#     valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
#     x_valid = x_array[valid_mask]
#     y_valid = y_array[valid_mask]
    
#     # Plot all points (including NaN)
#     plt.scatter(x_scores, first_order_coherence_score, alpha=0.7)
    
#     # Fit line only on valid data
#     if len(x_valid) > 1:  # Need at least 2 points for a line
#         m, b = np.polyfit(x_valid, y_valid, 1)
        
#         # Create line across the valid data range
#         x_line = np.linspace(min(x_valid), max(x_valid), 100)
#         y_line = m * x_line + b
#         plt.plot(x_line, y_line, color='red', linewidth=2, label='Line of Best Fit')
    
#     plt.xlabel(score_name.replace('_', ' ').title())
#     plt.ylabel('First Order Coherence Score')
#     plt.title(f'{score_name.replace("_", " ").title()} vs First Order Coherence Score')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'metric_plots/{score_name}_vs_first_order_coherence_score.png')
#     plt.close()


# # Plot vs second_order_coherence_score
# for score_name, x_scores in score_lists:
#     plt.figure()
    
#     # Filter out NaN values
#     x_array = np.array(x_scores)
#     y_array = np.array(second_order_coherence_score)
    
#     # Create mask for valid (non-NaN) values
#     valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
#     x_valid = x_array[valid_mask]
#     y_valid = y_array[valid_mask]
    
#     # Plot all points (including NaN)
#     plt.scatter(x_scores, second_order_coherence_score, alpha=0.7)
    
#     # Fit line only on valid data
#     if len(x_valid) > 1:  # Need at least 2 points for a line
#         m, b = np.polyfit(x_valid, y_valid, 1)
        
#         # Create line across the valid data range
#         x_line = np.linspace(min(x_valid), max(x_valid), 100)
#         y_line = m * x_line + b
#         plt.plot(x_line, y_line, color='red', linewidth=2, label='Line of Best Fit')
    
#     plt.xlabel(score_name.replace('_', ' ').title())
#     plt.ylabel('Second Order Coherence Score')
#     plt.title(f'{score_name.replace("_", " ").title()} vs Second Order Coherence Score')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'metric_plots/{score_name}_vs_second_order_coherence_score.png')
#     plt.close()