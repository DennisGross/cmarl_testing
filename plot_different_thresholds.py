import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Load the data from the CSV file
file_path = 'logs/central_logs.csv'
data = pd.read_csv(file_path)
# Filter only env_name = 'simple_spread_v3'
data = data[data['env_name'] == 'Simple Spread']
# model_path = "central_policies/simple_spread_v3_2/model.zip"
data = data[data['model_path'] == 'central_policies/simple_spread_v3_2/model.zip']
# From these, only take local_ratio=0
data = data[data['local_ratio'] == 0]


# Group by 'test_method' and 'top_percentage', then calculate the mean of 'avg_reward'
grouped_data = data.groupby(['test_method', 'top_percentage'])['avg_reward'].mean().unstack('test_method')
# Calculate Spearman correlation between each test method's 'top_percentage' and 'avg_reward'
correlations = {}
for method in grouped_data.columns:
    correlation, _ = spearmanr(grouped_data.index, grouped_data[method])
    correlations[method] = correlation

# Plot the data
plt.figure(figsize=(10,5.8))
# Increase font size
plt.rcParams.update({'font.size': 17})
plt.plot(grouped_data.index, grouped_data['Random Testing'], label='Random Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Random DisQ Testing'], label='Random DisQ Testing', marker='o')
#plt.plot(grouped_data.index, grouped_data['Random Q-low Testing'], label='Random Q-low Testing', marker='o')

plt.plot(grouped_data.index, grouped_data['Genetic Testing'], label='Genetic Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic DisQ Testing'], label='Genetic DisQ Testing', marker='o')


# Add labels and title
plt.xlabel('Top Percentage')
plt.ylabel('Average Reward')
#plt.title('Average Reward by Top Percentage and Test Method')
# Add legend
plt.legend()
plt.grid()
# Display the plot
plt.savefig("plots/centralized_different_thresholds.png")
# eps format
plt.savefig("plots/centralized_different_thresholds.eps", format='eps')

print(correlations)




# DECENTRALIZED
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'logs/dec_logs.csv'
data = pd.read_csv(file_path)
# Filter only env_name = 'simple_spread_v3'
data = data[data['env_name'] == 'Simple Spread']
# model_path = "decentral_policies/simple_spread_v3_2"
data = data[data['model_path'] == 'decentral_policies/simple_spread_v3_2']
# From these, only take local_ratio=0
data = data[data['local_ratio'] == 0]


# Group by 'test_method' and 'top_percentage', then calculate the mean of 'avg_reward'
grouped_data = data.groupby(['test_method', 'top_percentage'])['avg_reward'].mean().unstack('test_method')
# Calculate Spearman correlation between each test method's 'top_percentage' and 'avg_reward'
correlations = {}
for method in grouped_data.columns:
    correlation, _ = spearmanr(grouped_data.index, grouped_data[method])
    correlations[method] = correlation
    


# Plot the data
plt.figure(figsize=(10,5.8))
plt.plot(grouped_data.index, grouped_data['Random Testing'], label='Random Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Random DisQ Testing'], label='Random DisQ Testing', marker='o')
#plt.plot(grouped_data.index, grouped_data['Random Q-low Testing'], label='Random Q-low Testing', marker='o')

plt.plot(grouped_data.index, grouped_data['Genetic Testing'], label='Genetic Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic DisQ Testing'], label='Genetic DisQ Testing', marker='o')


# Add labels and title
plt.xlabel('Top Percentage')
plt.ylabel('Average Reward')
plt.rcParams.update({'font.size': 17})
#plt.title('Average Reward by Top Percentage and Test Method')

# Add legend
plt.legend()
plt.grid()
# Display the plot
plt.savefig("plots/dec_different_thresholds.png")
# eps format
plt.savefig("plots/dec_different_thresholds.eps", format='eps')

print(correlations)