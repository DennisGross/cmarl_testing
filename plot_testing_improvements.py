import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('logs/central_logs.csv')
# Filter out Random Q-low Testing
df = df[df['test_method'] != 'Random Q-low Testing']
# Create advanced_env_name by combining env_name and N_agents
df['advanced_env_name'] = df['env_name'] + ' (' + df['N_agents'].astype(str) + ')'
# Replace Coin Collecotr (2) with 3
df['advanced_env_name'] = df['advanced_env_name'].replace('Coin Collector (2)', 'Coin Collector (3)')
# Replace Knight Wizard Zombies (2) with 3
df['advanced_env_name'] = df['advanced_env_name'].replace('Knight Wizard Zombies (2)', 'Knight Wizard Zombies (3)')
# Filter out local_ratio>0
df = df[df['local_ratio'] == 0]

# Get the unique environments and test methods
env_names = df['advanced_env_name'].unique()
test_methods = df['test_method'].unique()

# Initialize a list to hold the improvements
improvement_list = []

for env in env_names:
    # Get the baseline avg_reward for 'Random Testing' for this env
    baseline_reward = df[(df['advanced_env_name'] == env) & (df['test_method'] == 'Random Testing')]['avg_reward'].mean()
    
    # For each test method, calculate the improvement over the baseline
    improvements = {'advanced_env_name': env}
    for method in test_methods:
        method_reward = df[(df['advanced_env_name'] == env) & (df['test_method'] == method)]['avg_reward'].mean()
        print("Method: ", method, " Reward: ", method_reward, " Baseline: ", baseline_reward, " Improvement: ", ((method_reward - baseline_reward) / abs(baseline_reward)) * 100)
        improvement = ((method_reward - baseline_reward) / abs(baseline_reward)) * 100  # Improvement over baseline in percentage
        improvements[method] = improvement
    # Append the dictionary to the list
    improvement_list.append(improvements)

# Create the DataFrame from the list
improvement_df = pd.DataFrame(improvement_list)

# Now plot the data
# Set up the bar plot
x = np.arange(len(env_names))  # the label locations
width = 0.15  # the width of the bars
plt.rcParams.update({'font.size': 17})
fig, ax = plt.subplots(figsize=(12, 9))

for i, method in enumerate(test_methods):
    offsets = x + (i - len(test_methods)/2) * width + width/2
    y_values = improvement_df[method].astype(float)
    ax.bar(offsets, y_values, width, label=method)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Reward Change over Random Testing (%)')
#ax.set_title('Testing Method and Environment')
ax.set_xticks(x)
ax.set_xticklabels(env_names, rotation=45)
ax.legend()

# Optional: Add a horizontal line at y=0
ax.axhline(0, color='black', linewidth=0.8)

fig.tight_layout()
plt.grid()

plt.savefig('plots/central_testing_improvements.png')
# eps format
plt.savefig('plots/central_testing_improvements.eps', format='eps')





# DECENTRAL
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
df = pd.read_csv('logs/dec_logs.csv')
# Filter out Random Q-low Testing
df = df[df['test_method'] != 'Random Q-low Testing']
# Create advanced_env_name by combining env_name and N_agents
df['advanced_env_name'] = df['env_name'] + ' (' + df['N_agents'].astype(str) + ')'
# Replace Coin Collecotr (2) with 3
df['advanced_env_name'] = df['advanced_env_name'].replace('Coin Collector (2)', 'Coin Collector (3)')
# Replace Knight Wizard Zombies (2) with 3
df['advanced_env_name'] = df['advanced_env_name'].replace('Knight Wizard Zombies (2)', 'Knight Wizard Zombies (3)')
# Filter out local_ratio>0
df = df[df['local_ratio'] == 0]

# Get the unique environments and test methods
env_names = df['advanced_env_name'].unique()
test_methods = df['test_method'].unique()

# Initialize a list to hold the improvements
improvement_list = []

for env in env_names:
    # Get the baseline avg_reward for 'Random Testing' for this env
    baseline_reward = df[(df['advanced_env_name'] == env) & (df['test_method'] == 'Random Testing')]['avg_reward'].mean()
    
    # For each test method, calculate the improvement over the baseline
    improvements = {'advanced_env_name': env}
    for method in test_methods:
        method_reward = df[(df['advanced_env_name'] == env) & (df['test_method'] == method)]['avg_reward'].mean()
        print("Method: ", method, " Reward: ", method_reward, " Baseline: ", baseline_reward, " Improvement: ", ((method_reward - baseline_reward) / abs(baseline_reward)) * 100)
        improvement = ((method_reward - baseline_reward) / abs(baseline_reward)) * 100  # Improvement over baseline in percentage
        improvements[method] = improvement
    # Append the dictionary to the list
    improvement_list.append(improvements)

# Create the DataFrame from the list
improvement_df = pd.DataFrame(improvement_list)

# Now plot the data
# Set up the bar plot
x = np.arange(len(env_names))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 9))

for i, method in enumerate(test_methods):
    offsets = x + (i - len(test_methods)/2) * width + width/2
    y_values = improvement_df[method].astype(float)
    ax.bar(offsets, y_values, width, label=method)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Reward Change over Random Testing (%)')
#ax.set_title('Testing Method and Environment')
ax.set_xticks(x)
ax.set_xticklabels(env_names, rotation=45)
ax.legend()

# Optional: Add a horizontal line at y=0
ax.axhline(0, color='black', linewidth=0.8)

fig.tight_layout()
plt.grid()

plt.savefig('plots/decentral_testing_improvements.png')
# eps format
plt.savefig('plots/decentral_testing_improvements.eps', format='eps')

