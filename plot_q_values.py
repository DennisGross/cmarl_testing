import matplotlib.pyplot as plt
import numpy as np

# Q-values for both agents
q_values1 = [-43.2524, -45.0721, -42.3529, -42.9642, -45.0303]
q_values2 = [-38.9369, -40.7882, -37.8546, -39.0253, -40.1937]
# 4.498207092285156 3.792169189453126
# -83.06721228740273

# Action labels
actions = ['no_action', 'move_left', 'move_right', 'move_down', 'move_up']
x = np.arange(len(actions))

# Width of bars
width = 0.35

# Plotting the multibar plot
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, q_values1, width, label='Agent 1')
bars2 = ax.bar(x + width/2, q_values2, width, label='Agent 2')

# Labeling
ax.set_xlabel('Actions')
ax.set_ylabel('Q-values')
#ax.set_title('Q-values of Agents for Different Actions')
ax.set_xticks(x)
ax.set_xticklabels(actions)
# y-interval between 0 and -50
ax.set_ylim(-50, 0)
ax.legend()
plt.grid()
# Display plot
plt.savefig('q_values_faulty.png')
# Save as eps file
plt.savefig('q_values_faulty.eps')



import matplotlib.pyplot as plt
import numpy as np

# Q-values for both agents
q_values1 = [-85.4748, -88.3745, -84.0478, -87.2898, -86.2145]
q_values2 = [-94.1406, -96.1284, -93.6179, -96.8851, -94.1228]
#9.570068359375 6.789749145507813
#-107.28405586172246

# Action labels
actions = ['no_action', 'move_left', 'move_right', 'move_down', 'move_up']
x = np.arange(len(actions))

# Width of bars
width = 0.35

# Plotting the multibar plot
fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, q_values1, width, label='Agent 1')
bars2 = ax.bar(x + width/2, q_values2, width, label='Agent 2')

# Labeling
ax.set_xlabel('Actions')
ax.set_ylabel('Q-values')
#ax.set_title('Q-values of Agents for Different Actions')
ax.set_xticks(x)
ax.set_xticklabels(actions)
# y-interval between 0 and -50
ax.legend()
plt.grid()
# Display plot
plt.savefig('q_values_dec.png')
# Save as eps file
plt.savefig('q_values_dec.eps')