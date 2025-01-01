"""
t-SNE to cluster games based on team statistics and create a 
visualization that might reveal patterns in how games unfold. 
Color-code points based on whether the home team won or lost
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def aggregate_game_stats(players_df):
    # Group by game and team to get team-level stats for each game
    game_stats = players_df.groupby(['Game', 'Team']).agg({
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

    # Calculate percentages and additional metrics
    game_stats['FG_PCT'] = game_stats['FGM'] / game_stats['FGA']
    game_stats['FG3_PCT'] = game_stats['FG3M'] / game_stats['FG3A']
    game_stats['FT_PCT'] = game_stats['FTM'] / game_stats['FTA']
    game_stats['TOTAL_REB'] = game_stats['ORB'] + game_stats['DRB']

    return game_stats


# players_df = pd.read_csv(r"D:\_FEL\SAN\project\DataDribbling\data3-07-09\players.csv")
players_df = pd.read_csv('merged_players.csv')

# Aggregate stats
game_stats = aggregate_game_stats(players_df)

# Create features for each game by combining home and away team stats
game_features = []
for game_id in game_stats['Game'].unique():
    game_data = game_stats[game_stats['Game'] == game_id]

    if len(game_data) != 2:  # Skip games without exactly 2 teams
        continue

    # Sort by Team ID to ensure consistent home/away ordering
    game_data = game_data.sort_values('Team')

    # Create feature dictionary for this game
    game_dict = {
        'Game': game_id,
        'Home_Team': game_data.iloc[0]['Team'],
        'Away_Team': game_data.iloc[1]['Team'],
        'Home_Points': game_data.iloc[0]['PTS'],
        'Away_Points': game_data.iloc[1]['PTS']
    }

    # Add home team stats with 'H_' prefix
    for col in ['FG_PCT', 'FG3_PCT', 'FT_PCT', 'ORB', 'DRB', 'TOTAL_REB', 'AST', 'STL', 'BLK', 'TOV', 'PF']:
        game_dict[f'H_{col}'] = game_data.iloc[0][col]
        game_dict[f'A_{col}'] = game_data.iloc[1][col]

    game_features.append(game_dict)

# Convert to DataFrame
games_df = pd.DataFrame(game_features)

# Create target variable (1 if home team won, 0 if lost)
games_df['Home_Win'] = (games_df['Home_Points'] > games_df['Away_Points']).astype(int)

# Select features for t-SNE
feature_cols = [col for col in games_df.columns if col.startswith(('H_', 'A_'))]
X = games_df[feature_cols]

# Handle any NaN values
X = X.fillna(X.mean())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=100)
X_tsne = tsne.fit_transform(X_scaled)

# Create visualization
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=games_df['Home_Win'],
                      cmap='coolwarm',
                      alpha=0.6)

plt.colorbar(scatter).set_label(label='Home Team Loss (0) / Win (1)', size=15)
# plt.title('t-SNE Visualization of Game Patterns')
plt.xlabel('t-SNE component 1', fontsize=15)
plt.ylabel('t-SNE component 2', fontsize=15)

# Add point score difference as annotations for some interesting points
for idx in range(len(X_tsne)):
    score_diff = games_df.iloc[idx]['Home_Points'] - games_df.iloc[idx]['Away_Points']
    # Show top 10% most extreme games
    if abs(score_diff) > np.percentile(abs(games_df['Home_Points'] - games_df['Away_Points']), 99):
        plt.annotate(f'{score_diff:+}',
                     (X_tsne[idx, 0], X_tsne[idx, 1]),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=9)

plt.grid(True, alpha=0.3)
plt.show()

X_2d = X_tsne  # shape (n_samples, 2)

y = games_df['Home_Win']

X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.3, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Přesnost separace výher/proher na základě t-SNE: {accuracy:.2f}")