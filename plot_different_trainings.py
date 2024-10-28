import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'logs/central_logs.csv'
data = pd.read_csv(file_path)
data = data[data['model_path'].str.contains('simple_spread_v3_2_t')]
# Filter only env_name = 'simple_spread_v3'
data = data[data['env_name'] == 'Simple Spread']



# Group by 'test_method' and 'top_percentage', then calculate the mean of 'avg_reward'
grouped_data = data.groupby(['test_method', 'model_path'])['avg_reward'].mean().unstack('test_method')
print(grouped_data)
# Add ticks for the x-axis
grouped_data.index = grouped_data.index.str.replace('central_policies/simple_spread_v3_2_t/', '')
# Plot the data
plt.figure(figsize=(10,5.8))
# Increase font size
plt.rcParams.update({'font.size': 17})
plt.plot(grouped_data.index, grouped_data['Random Q-testing'], label='Random Q-testing', marker='o')
#plt.plot(grouped_data.index, grouped_data['Random Q-low-testing'], label='Random Q-low-testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Random Testing'], label='Random Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic Testing'], label='Genetic Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic Q-testing'], label='Genetic Q-testing', marker='o')


# Add labels and title
plt.xlabel('Top Percentage')
plt.ylabel('Average Reward')
#plt.title('Average Reward by Top Percentage and Test Method')
# Add legend
plt.legend()
plt.grid()
# Display the plot
plt.savefig("plots/centralized_different_trainings.png")
# eps format
plt.savefig("plots/centralized_different_trainings.eps", format='eps')


# DECENTRALIZED
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
file_path = 'logs/dec_logs.csv'
data = pd.read_csv(file_path)
data = data[data['model_path'].str.contains('simple_spread_v3_2_t')]
# Filter only env_name = 'simple_spread_v3'
data = data[data['env_name'] == 'Simple Spread']
# From these, only take local_ratio=0
data = data[data['local_ratio'] == 0]


# Group by 'test_method' and 'top_percentage', then calculate the mean of 'avg_reward'
grouped_data = data.groupby(['test_method', 'top_percentage'])['avg_reward'].mean().unstack('test_method')

# Plot the data
plt.figure(figsize=(10,5.8))
plt.plot(grouped_data.index, grouped_data['Random Q-testing'], label='Random Q-testing', marker='o')
#plt.plot(grouped_data.index, grouped_data['Random Q-low-testing'], label='Random Q-low-testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Random Testing'], label='Random Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic Testing'], label='Genetic Testing', marker='o')
plt.plot(grouped_data.index, grouped_data['Genetic Q-testing'], label='Genetic Q-testing', marker='o')


# Add labels and title
plt.xlabel('Top Percentage')
plt.ylabel('Average Reward')
plt.rcParams.update({'font.size': 17})
#plt.title('Average Reward by Top Percentage and Test Method')

# Add legend
plt.legend()
plt.grid()
# Display the plot
plt.savefig("plots/dec_different_trainings.png")
# eps format
plt.savefig("plots/dec_different_trainings.eps", format='eps')