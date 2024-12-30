import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Function to aggregate player statistics


def aggregate_player_stats(df):
    # Calculate per-game averages for each player
    player_stats = df.groupby('Player').agg({
        'MIN': 'mean',
        'PTS': 'mean',
        'RB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean',
        'TOV': 'mean',
        'FGA': 'mean',
        'FGM': 'mean',
        'FG3A': 'mean',
        'FG3M': 'mean',
        'FTA': 'mean',
        'FTM': 'mean'
    }).reset_index()

    # Calculate additional metrics
    player_stats['FG_PCT'] = (player_stats['FGM'] / player_stats['FGA']).fillna(0)
    player_stats['FG3_PCT'] = (player_stats['FG3M'] / player_stats['FG3A']).fillna(0)
    player_stats['FT_PCT'] = (player_stats['FTM'] / player_stats['FTA']).fillna(0)

    return player_stats


def perform_pca_analysis(stats_df, features):
    # Extract features for PCA
    X = stats_df[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    return X_pca, pca, explained_variance_ratio


def plot_pca_results(X_pca, pca, stats_df, features):
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))

    # Plot 1: Scatter plot of first two principal components
    ax1 = fig.add_subplot(121)
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=stats_df['PTS'], cmap='viridis',
                          alpha=0.6)
    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.set_title('Player Distribution in PCA Space')
    plt.colorbar(scatter, label='Points per Game')

    # Plot 2: Feature importance in first two PCs
    ax2 = fig.add_subplot(122)
    components = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(features))],
        index=features
    )
    sns.heatmap(components[['PC1', 'PC2']], cmap='RdBu', center=0, ax=ax2)
    ax2.set_title('Feature Importance in First Two Principal Components')

    plt.tight_layout()
    plt.show()
    return fig


def print_pca_analysis(pca, features, explained_variance_ratio):
    print("=== PCA Analysis Results ===\n")

    # Print explained variance
    print("Explained Variance Ratio by Component:")
    for i, ratio in enumerate(explained_variance_ratio, 1):
        print(f"PC{i}: {ratio:.3f} ({ratio*100:.1f}%)")
    print(f"\nCumulative Variance Explained: {np.sum(explained_variance_ratio)*100:.1f}%\n")

    # Print feature importance for first two components
    print("Feature Importance in Principal Components:")
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(features))],
        index=features
    )
    print("\nPC1 Feature Contributions:")
    for feature, value in components_df['PC1'].sort_values(ascending=False).items():
        print(f"{feature}: {value:.3f}")

    print("\nPC2 Feature Contributions:")
    for feature, value in components_df['PC2'].sort_values(ascending=False).items():
        print(f"{feature}: {value:.3f}")


def analyze_player_archetypes(players_df):
    # Aggregate player statistics
    player_stats = aggregate_player_stats(players_df)

    # Define features for PCA
    features = ['MIN', 'PTS', 'RB', 'AST', 'STL', 'BLK', 'TOV',
                'FG_PCT', 'FG3_PCT', 'FT_PCT']

    # Perform PCA
    X_pca, pca, explained_variance = perform_pca_analysis(player_stats, features)
    print_pca_analysis(pca, features, explained_variance)
    """
    High positive values for BLK, FG_PCT, and RB mean this component strongly represents players who 
    are good at blocking, efficient shooting, and rebounding
    High negative values for AST and FG3_PCT suggest it contrasts with playmaking and three-point shooting
    """

    # Create visualization
    fig = plot_pca_results(X_pca, pca, player_stats, features)

    return player_stats, X_pca, pca, explained_variance, fig


players_df = pd.read_csv('players.csv')
player_stats, X_pca, pca, explained_variance, fig = analyze_player_archetypes(players_df)
