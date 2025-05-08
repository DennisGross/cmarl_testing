import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file without headers (unnamed columns)
# Replace 'your_file.csv' with the actual file path
file_path = 'logs/central_all_rewards.csv'
df = pd.read_csv(file_path, header=None, sep=',')
# Filter out local_ratio>0
#df = df[df['local_ratio'] == 0]


# Rename the columns for clarity
df.columns = ['model_path', 'testing_method', 'reward']

# Filter out rows with testing_method Q-Low-Testing
# Rename Q-low-Testing to Q-low Testing
df['testing_method'] = df['testing_method'].replace('Random Q-low-testing', 'Random Q-low Testing')
# Filter out Random Q-low Testing
df = df[df['testing_method'] != 'Random Q-low Testing']
# Rename Q-Testing to DisQ Testing
df['testing_method'] = df['testing_method'].replace('Random Q-testing', 'Random DisQ Testing')
# Rename Genetic Q-Testing to Genetic DisQ Testing
df['testing_method'] = df['testing_method'].replace('Genetic Q-testing', 'Genetic DisQ Testing')

# Filter out rows where model_path is 'central_policies/simple_speaker_listener_v4/model.zip'
df_filtered = df[df['model_path'] == 'central_policies/simple_speaker_listener_v4/model.zip']

# Group by 'testing_method' and collect rewards for each group
grouped_data = df_filtered.groupby('testing_method')['reward'].apply(list)

# Create the boxplot without model_path
plt.figure(figsize=(10, 5.2))
plt.rcParams.update({'font.size': 17})
plt.boxplot(grouped_data, labels=grouped_data.index, patch_artist=True)
#plt.title('Rewards for Simple Speaker Listener v4')
plt.xlabel('Testing Method')
plt.ylabel('Reward')
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.tight_layout()
plt.grid()

# Show the plot
plt.savefig('plots/reward_boxplot_central.png')
# Save as eps
plt.savefig('plots/reward_boxplot_central.eps')

# DECENTRAL
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file without headers (unnamed columns)
# Replace 'your_file.csv' with the actual file path
file_path = 'logs/dec_all_rewards.csv'
df = pd.read_csv(file_path, header=None, sep=',')
# Filter out local_ratio>0
#df = df[df['local_ratio'] == 0]


# Rename the columns for clarity
df.columns = ['model_path', 'testing_method', 'reward']

# Filter out rows with testing_method Q-Low-Testing
#df_filtered = df[df['testing_method'] != 'Q-Low-Testing']
# Rename Q-low-Testing to Q-low Testing
df['testing_method'] = df['testing_method'].replace('Random Q-low-testing', 'Random Q-low Testing')
# Filter out Random Q-low Testing
df = df[df['testing_method'] != 'Random Q-low Testing']
# Rename Q-Testing to DisQ Testing
df['testing_method'] = df['testing_method'].replace('Random Q-testing', 'Random DisQ Testing')
# Rename Genetic Q-Testing to Genetic DisQ Testing
df['testing_method'] = df['testing_method'].replace('Genetic Q-testing', 'Genetic DisQ Testing')

# Filter out rows where model_path is 'decentral_policies/simple_speaker_listener_v4'
df_filtered = df[df['model_path'] == 'decentral_policies/simple_speaker_listener_v4']

# Group by 'testing_method' and collect rewards for each group
grouped_data = df_filtered.groupby('testing_method')['reward'].apply(list)

# Create the boxplot without model_path
plt.figure(figsize=(10, 5.2))
plt.rcParams.update({'font.size': 17})
plt.boxplot(grouped_data, labels=grouped_data.index, patch_artist=True)
#plt.title('Rewards for Simple Speaker Listener v4')
plt.xlabel('Testing Method')
plt.ylabel('Reward')
plt.xticks(rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.tight_layout()
plt.grid()
# Show the plot
plt.savefig('plots/reward_boxplot_decentral.png')
# eps
plt.savefig('plots/reward_boxplot_decentral.eps')
