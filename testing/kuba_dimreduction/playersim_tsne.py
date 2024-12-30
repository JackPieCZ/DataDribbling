import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# import seaborn as sns


def prepare_player_data(players_df):
    # Aggregate player statistics
    player_stats = players_df.groupby('Player').agg({
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

    # Calculate efficiency metrics
    player_stats['FG_PCT'] = (player_stats['FGM'] / player_stats['FGA']).fillna(0)
    player_stats['FG3_PCT'] = (player_stats['FG3M'] / player_stats['FG3A']).fillna(0)
    player_stats['FT_PCT'] = (player_stats['FTM'] / player_stats['FTA']).fillna(0)

    return player_stats


def perform_tsne_analysis(data, features, perplexities=[5, 30, 50, 100], random_state=42):
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[features])

    # Perform t-SNE for different perplexity values
    tsne_results = {}
    for perp in perplexities:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
        tsne_result = tsne.fit_transform(scaled_data)
        tsne_results[perp] = tsne_result

    return tsne_results, scaled_data


def plot_tsne_results(tsne_results, player_stats, color_feature='PTS'):
    # Create a figure with subplots for each perplexity value
    n_plots = len(tsne_results)
    fig = plt.figure(figsize=(20, 5 * ((n_plots + 1) // 2)))

    for idx, (perp, embedding) in enumerate(tsne_results.items(), 1):
        ax = fig.add_subplot(((n_plots + 1) // 2), 2, idx)

        scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                             c=player_stats[color_feature],
                             cmap='viridis',
                             alpha=0.6)

        ax.set_title(f't-SNE with Perplexity {perp}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=ax, label=color_feature)

    plt.tight_layout()
    return fig


def analyze_clusters(tsne_results, player_stats, features):
    # Calculate average distance to nearest neighbors for each perplexity
    distances = {}
    for perp, embedding in tsne_results.items():
        # Calculate pairwise distances
        dist_matrix = np.sqrt(((embedding[:, None, :] - embedding) ** 2).sum(axis=2))
        # Get average distance to 5 nearest neighbors (excluding self)
        sorted_dist = np.sort(dist_matrix, axis=1)
        avg_nearest_dist = np.mean(sorted_dist[:, 1:6])  # exclude self (0th neighbor)
        distances[perp] = avg_nearest_dist

    print("\n=== t-SNE Analysis Results ===")
    print("\nAverage distance to 5 nearest neighbors:")
    for perp, dist in distances.items():
        print(f"Perplexity {perp}: {dist:.2f}")

    # Find correlations between original features and t-SNE dimensions
    print("\nFeature correlations with t-SNE dimensions:")
    for perp, embedding in tsne_results.items():
        print(f"\nPerplexity {perp}:")
        correlations = {}
        for feature in features:
            corr_dim1 = np.corrcoef(player_stats[feature], embedding[:, 0])[0, 1]
            corr_dim2 = np.corrcoef(player_stats[feature], embedding[:, 1])[0, 1]
            correlations[feature] = (corr_dim1, corr_dim2)

        # Sort by absolute correlation strength
        sorted_correlations = sorted(correlations.items(),
                                     key=lambda x: max(abs(x[1][0]), abs(x[1][1])),
                                     reverse=True)

        for feature, (corr1, corr2) in sorted_correlations[:5]:  # Show top 5 correlations
            print(f"{feature}: Dim1={corr1:.3f}, Dim2={corr2:.3f}")


def analyze_player_similarity(players_df):
    # Prepare data
    player_stats = prepare_player_data(players_df)

    # Define features for analysis
    features = ['MIN', 'PTS', 'RB', 'AST', 'STL', 'BLK', 'TOV',
                'FG_PCT', 'FG3_PCT', 'FT_PCT']

    # Perform t-SNE analysis
    perplexities = [5, 30, 50, 100]
    tsne_results, scaled_data = perform_tsne_analysis(player_stats, features, perplexities)

    # Create visualizations
    fig_points = plot_tsne_results(tsne_results, player_stats, 'PTS')
    fig_assists = plot_tsne_results(tsne_results, player_stats, 'AST')

    # Analyze cluster structure
    analyze_clusters(tsne_results, player_stats, features)

    return player_stats, tsne_results, scaled_data, (fig_points, fig_assists)


players_df = pd.read_csv('players.csv')
player_stats, tsne_results, scaled_data, figures = analyze_player_similarity(players_df)
plt.show()