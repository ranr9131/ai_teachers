import json
import matplotlib.pyplot as plt


with open('data/connectedness_scores.json', 'r') as f:
    data = json.load(f)

scores = list(data.values())

num_sentences = [score[0] for score in scores]
num_clusters = [score[1] for score in scores]


plt.scatter(num_sentences, num_clusters)
plt.xlabel('Number of Sentences')
plt.ylabel('Number of Clusters')
plt.title('Scatter Plot of Sentences vs Clusters')

# Add a dotted line y = x
min_val = min(min(num_sentences), min(num_clusters))
max_val = max(max(num_sentences), max(num_clusters))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y = x')

plt.legend()
plt.savefig("test.png")