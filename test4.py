import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

with open('data/connectedness_scores.json', 'r') as f:
    data = json.load(f)

scores = list(data.values())
connectedness = [score[2] for score in scores]

years_experience = [12, 22, 7, 21, 13, 25, 6, 12, 3, 25, 17, 8, 14, 11, 22, 2, 29, 12, 8, 20, 9, 2, 27, 3]

output_productivity = [0, 3, 2, 3, 5, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 5, 0, 0, 5, 2, 4, 0, 4]

clear_goals = [
    "Did not respond",
    "Had clear goals",
    "Had clear goals",
    "Had clear goals",
    "Had clear goals",
    "No clear goals",
    "Some ideas",
    "Some ideas",
    "Did not respond",
    "Some ideas",
    "Had clear goals",
    "Had clear goals",
    "Did not respond",
    "Some ideas",
    "No clear goals",
    "Did not respond",
    "Had clear goals",
    "Did not respond",
    "Did not respond",
    "No clear goals",
    "Some ideas",
    "Some ideas",
    "Did not respond",
    "Had clear goals"
]

df = pd.DataFrame({
    'connectedness': connectedness,
    'years_experience': years_experience,
    "output_productivity": output_productivity,
    'clear_goals': clear_goals
})

seq1 = [4, 7, 10, 13, 20, 21]
seq2 = [1, 3, 8, 9, 18]

sum = 0
count = 0
for teachID in seq1:
    if output_productivity[teachID-1] != 0:
        sum += output_productivity[teachID-1]
        count += 1

print(f"Sequence 1 teacher average output productivity: {sum/count}")

sum = 0
count = 0
for teachID in seq2:
    if output_productivity[teachID-1] != 0:
        sum += output_productivity[teachID-1]
        count += 1

print(f"Sequence 2 teacher average output productivity: {sum/count}")


sum = 0
count = 0
for teachID in [i for i in range(1,25)]:
    if output_productivity[teachID-1] != 0:
        sum += output_productivity[teachID-1]
        count += 1

print(f"all teacher average output productivity: {sum/count}")




# import matplotlib.pyplot as plt

# markers = {
#     "Did not respond": {"marker": "x", "color": "gray"},
#     "Had clear goals": {"marker": "o", "color": "blue"},
#     "No clear goals": {"marker": "^", "color": "red"},
#     "Some ideas": {"marker": "D", "color": "green"}
# }

# for label, props in markers.items():
#     subset = df[df['clear_goals'] == label]
#     plt.scatter(
#         subset['connectedness'],
#         subset['output_productivity'],
#         label=label,
#         marker=props["marker"],
#         color=props["color"],
#         s=100
#     )

# # Line of best fit (ignoring "Did not respond")
# fit_df = df[df['clear_goals'] != "Did not respond"]
# x = fit_df['connectedness']
# y = fit_df['output_productivity']
# m, b = np.polyfit(x, y, 1)
# plt.plot(x, m*x + b, color='black', linestyle='--', label='Best Fit')

# # Pearson r coefficient
# r, _ = pearsonr(x, y)
# print(f'Pearson r = {r:.2f}')

# plt.xlabel('Connectedness')
# plt.ylabel('Output Productivity')
# plt.legend(title='Clear Goals')
# plt.title('Connectedness vs Output Productivity')
# plt.savefig("test.png")


