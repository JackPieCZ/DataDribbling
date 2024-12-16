import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


data = pd.read_csv('./data/season_1_matches.csv') 
# Create a DataFrame
df = pd.DataFrame(data)


# Map 'W' to colors: 1 -> Green (Win), 0 -> Red (Loss)
colors = df['W'].map({1: 'green', 0: 'red'})

# Map Home/Away ('H'/'A') to Z-axis values: H -> 1, A -> 0
df['Z'] = df['Team'].map({'H': 1, 'A': 0})

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points in 3D
ax.scatter(df['SC'], df['ACC'], df['RB'], c=colors, s=25, edgecolor='black', alpha=0.7)

# Set axis labels
ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_zlabel('Number of rebounds', fontsize=12)
ax.set_title('3D Visualization of Team Performance (Home/Away)', fontsize=14)

# Add a legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='Win', markerfacecolor='green', markersize=10),
    plt.Line2D([0], [0], marker='o', color='w', label='Loss', markerfacecolor='red', markersize=10)
]
ax.legend(handles=legend_elements, loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()