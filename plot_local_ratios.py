import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('logs/central_local_ratio.csv')

data['test_method'] = data['test_method'].replace('Random Q-low-testing', 'Random Q-low Testing')
# Filter out Random Q-low Testing
data = data[data['test_method'] != 'Random Q-low Testing']
# Rename Q-Testing to DisQ Testing
data['test_method'] = data['test_method'].replace('Random Q-testing', 'Random DisQ Testing')
# Rename Genetic Q-Testing to Genetic DisQ Testing
data['test_method'] = data['test_method'].replace('Genetic Q-testing', 'Genetic DisQ Testing')

# Get the unique testing methods
testing_methods = data['test_method'].unique()



# Sort the local ratios
local_ratios = sorted(data['local_ratio'].unique())

# Initialize the plot
plt.figure(figsize=(10, 5.6))
plt.rcParams.update({'font.size': 17})
# Plot avg_reward vs local_ratio for each testing method
for method in testing_methods:
    # Filter data for the current testing method
    method_data = data[data['test_method'] == method]
    # Sort data by local_ratio
    method_data = method_data.sort_values('local_ratio')
    
    # Plot the data
    plt.plot(
        method_data['local_ratio'],
        method_data['avg_reward'],
        marker='o',
        label=method
    )

# Add labels and title
plt.xlabel('Local Ratio')
plt.ylabel('Average Reward')
#plt.title('Average Reward vs. Local Ratio for Different Testing Methods')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("plots/central_local_ratio.png")
# Save as eps
plt.savefig("plots/central_local_ratio.eps")
# DECENTRAL

import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('logs/dec_local_ratio.csv')

data['test_method'] = data['test_method'].replace('Random Q-low-testing', 'Random Q-low Testing')
# Filter out Random Q-low Testing
data = data[data['test_method'] != 'Random Q-low Testing']
# Rename Q-Testing to DisQ Testing
data['test_method'] = data['test_method'].replace('Random Q-testing', 'Random DisQ Testing')
# Rename Genetic Q-Testing to Genetic DisQ Testing
data['test_method'] = data['test_method'].replace('Genetic Q-testing', 'Genetic DisQ Testing')

# Get the unique testing methods
testing_methods = data['test_method'].unique()



# Sort the local ratios
local_ratios = sorted(data['local_ratio'].unique())

# Initialize the plot
plt.figure(figsize=(10, 5.6))
plt.rcParams.update({'font.size': 17})
# Plot avg_reward vs local_ratio for each testing method
for method in testing_methods:
    # Filter data for the current testing method
    method_data = data[data['test_method'] == method]
    # Sort data by local_ratio
    method_data = method_data.sort_values('local_ratio')
    # Plot the data
    plt.plot(
        method_data['local_ratio'],
        method_data['avg_reward'],
        marker='o',
        label=method
    )

# Add labels and title
plt.xlabel('Local Ratio')
plt.ylabel('Average Reward')
#plt.title('Average Reward vs. Local Ratio for Different Testing Methods')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig("plots/dec_local_ratio.png")
# Save as eps
plt.savefig("plots/dec_local_ratio.eps")
