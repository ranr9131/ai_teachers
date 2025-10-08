from metrics import *
import pandas as pd
import altair as alt
import numpy as np


df = pd.read_csv('data/prompts_responses.csv')
df = df[["Original Prompt", "Output #1 (Andrea)"]]
df = df.rename(columns={"Output #1 (Andrea)": "Response", "Original Prompt": "Prompt"})


prompt_scores = []
response_scores = []

for i in range(len(df)):
    print(i)
    prompt_scores.append(coherence_score(df["Prompt"][i]))
    response_scores.append(response_depth_score(df["Response"][i]))


print(prompt_scores)
print(response_scores)


plot_df = pd.DataFrame({
    "Prompt Score": prompt_scores,
    "Response Score": response_scores
})

# Scatter plot
scatter = alt.Chart(plot_df).mark_point().encode(
    x="Prompt Score",
    y="Response Score"
)

# Line of best fit (regression line)
regression = scatter.transform_regression(
    "Prompt Score", "Response Score"
).mark_line(color='red')

chart = (scatter + regression).properties(
    title="Prompt Scores vs Response Scores"
)

chart.save('plots/prompt_vs_response2.png', scale_factor=2)