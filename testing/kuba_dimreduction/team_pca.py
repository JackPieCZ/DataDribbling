"""
PCA analysis of team performance metrics (scoring, rebounds, assists, etc.) 
to identify the main playing styles and patterns among teams. 
Visualize the results using a scatter plot with team IDs, and analyze which statistical categories 
contribute most to the principal components
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import seaborn as sns

# Read the data
players_df = pd.read_csv('./players.csv')

# Aggregate statistics by team
team_stats = players_df.groupby('Team').agg({
    'FGM': 'sum',
    'FGA': 'sum',
    'FG3M': 'sum',
    'FG3A': 'sum',
    'FTM': 'sum',
    'FTA': 'sum',
    'ORB': 'sum',
    'DRB': 'sum',
    'AST': 'sum',
    'STL': 'sum',
    'BLK': 'sum',
    'TOV': 'sum',
    'PF': 'sum',
    'PTS': 'sum'
}).reset_index()

# Calculate additional metrics
team_stats['FG_PCT'] = team_stats['FGM'] / team_stats['FGA']
team_stats['FG3_PCT'] = team_stats['FG3M'] / team_stats['FG3A']
team_stats['FT_PCT'] = team_stats['FTM'] / team_stats['FTA']
team_stats['TOTAL_REB'] = team_stats['ORB'] + team_stats['DRB']

# Select features for PCA
features = ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'ORB', 'DRB', 'TOTAL_REB',
            'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']

# Prepare the data
X = team_stats[features]

# Handle any NaN values
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Create scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

# Plot first two principal components
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.8)
for i, team in enumerate(team_stats['Team']):
    plt.annotate(team, (X_pca[i, 0], X_pca[i, 1]))
plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
plt.title('Team Performance PCA Analysis')
plt.grid(True)
plt.show()

# Create component loadings heatmap
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(features))],
    index=features
)
# plt.figure(figsize=(12, 8))
# sns.heatmap(loadings, annot=True, cmap='RdBu', center=0)
# plt.title('PCA Component Loadings')
# plt.tight_layout()
# plt.show()

# Print explained variance ratios
print("\nExplained variance ratio for each component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.3f}")

# Print feature contributions to first two components
print("\nTop contributing features to first two principal components:")
pc1_loadings = pd.Series(pca.components_[0], index=features).abs().sort_values(ascending=False)
pc2_loadings = pd.Series(pca.components_[1], index=features).abs().sort_values(ascending=False)

print("\nPC1 loadings:")
print(pc1_loadings)
print("\nPC2 loadings:")
print(pc2_loadings)
