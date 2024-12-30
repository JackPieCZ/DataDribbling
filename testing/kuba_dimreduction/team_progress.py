import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# import seaborn as sns


def prepare_team_stats(games_df, window_size=5):
    """
    Prepare rolling team statistics from game data
    """
    # Create home team stats
    home_stats = games_df[[
        'Date', 'HID', 'HSC', 'HFGM', 'HFGA', 'HFG3M', 'HFG3A', 'HFTM', 'HFTA',
        'HORB', 'HDRB', 'HAST', 'HSTL', 'HBLK', 'HTOV'
    ]].copy()

    # Rename columns to remove H prefix
    home_stats.columns = ['Date', 'TeamID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A',
                          'FTM', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV']

    # Create away team stats
    away_stats = games_df[[
        'Date', 'AID', 'ASC', 'AFGM', 'AFGA', 'AFG3M', 'AFG3A', 'AFTM', 'AFTA',
        'AORB', 'ADRB', 'AAST', 'ASTL', 'ABLK', 'ATOV'
    ]].copy()

    # Rename columns to remove A prefix
    away_stats.columns = ['Date', 'TeamID', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A',
                          'FTM', 'FTA', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'TOV']

    # Combine home and away stats
    all_stats = pd.concat([home_stats, away_stats])
    all_stats['Date'] = pd.to_datetime(all_stats['Date'])
    all_stats = all_stats.sort_values(['TeamID', 'Date'])

    # Calculate additional metrics
    all_stats['FG_PCT'] = (all_stats['FGM'] / all_stats['FGA']).fillna(0)
    all_stats['FG3_PCT'] = (all_stats['FG3M'] / all_stats['FG3A']).fillna(0)
    all_stats['FT_PCT'] = (all_stats['FTM'] / all_stats['FTA']).fillna(0)
    all_stats['RB'] = all_stats['ORB'] + all_stats['DRB']

    # Calculate rolling averages
    features = ['PTS', 'RB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    rolling_stats = all_stats.groupby('TeamID')[features].rolling(
        window=window_size, min_periods=1
    ).mean().reset_index()

    return rolling_stats


def perform_pca_trajectory_analysis(rolling_stats, features):
    """
    Perform PCA on rolling team statistics
    """
    # Prepare data for PCA
    X = rolling_stats[features]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Add PCA coordinates to the dataframe
    rolling_stats['PC1'] = X_pca[:, 0]
    rolling_stats['PC2'] = X_pca[:, 1]

    return rolling_stats, pca


def plot_team_trajectories(rolling_stats, pca, features, n_teams=None):
    """
    Create visualizations of team trajectories in PCA space
    """
    # Create figure for trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Get list of teams to plot (either all or specified number)
    teams = rolling_stats['TeamID'].unique()
    if n_teams is not None:
        teams = teams[:n_teams]

    # Plot trajectories
    for team in teams:
        team_data = rolling_stats[rolling_stats['TeamID'] == team]
        ax1.plot(team_data['PC1'], team_data['PC2'], '-o', label=f'Team {team}',
                 alpha=0.6, markersize=4)

        # Add arrow to show direction
        if len(team_data) > 1:
            ax1.arrow(team_data['PC1'].iloc[-2], team_data['PC2'].iloc[-2],
                      team_data['PC1'].iloc[-1] - team_data['PC1'].iloc[-2],
                      team_data['PC2'].iloc[-1] - team_data['PC2'].iloc[-2],
                      head_width=0.1, head_length=0.1, fc='k', ec='k', alpha=0.3)

    ax1.set_xlabel('First Principal Component')
    ax1.set_ylabel('Second Principal Component')
    ax1.set_title('Team Performance Trajectories')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot feature importance
    # components = pd.DataFrame(
    #     pca.components_.T,
    #     columns=['PC1', 'PC2'],
    #     index=features
    # )
    # sns.heatmap(components, cmap='RdBu', center=0, ax=ax2)
    # ax2.set_title('Feature Importance in Principal Components')

    # plt.tight_layout()
    return fig


def analyze_trajectory_patterns(rolling_stats):
    """
    Analyze patterns in team trajectories
    """
    # Calculate trajectory statistics
    trajectory_stats = rolling_stats.groupby('TeamID').agg({
        'PC1': ['mean', 'std', lambda x: x.iloc[-1] - x.iloc[0]],
        'PC2': ['mean', 'std', lambda x: x.iloc[-1] - x.iloc[0]]
    }).round(3)

    trajectory_stats.columns = ['PC1_Mean', 'PC1_Std', 'PC1_Change',
                                'PC2_Mean', 'PC2_Std', 'PC2_Change']

    print("\n=== Team Trajectory Analysis ===")
    print("\nTeams with largest PC1 movement:")
    print(trajectory_stats.nlargest(5, 'PC1_Change')[['PC1_Change']])

    print("\nTeams with largest PC2 movement:")
    print(trajectory_stats.nlargest(5, 'PC2_Change')[['PC2_Change']])

    print("\nTeams with most variable performance (highest std dev):")
    print(trajectory_stats.nlargest(5, 'PC1_Std')[['PC1_Std', 'PC2_Std']])

    return trajectory_stats


def analyze_team_evolution(games_df, window_size=5):
    """
    Main function to analyze team performance evolution
    """
    # Define features for analysis
    features = ['PTS', 'RB', 'AST', 'STL', 'BLK', 'TOV',
                'FG_PCT', 'FG3_PCT', 'FT_PCT']

    # Prepare rolling statistics
    rolling_stats = prepare_team_stats(games_df, window_size)

    # Perform PCA
    rolling_stats, pca = perform_pca_trajectory_analysis(rolling_stats, features)

    # Create visualization
    fig = plot_team_trajectories(rolling_stats, pca, features)

    # Analyze patterns
    trajectory_stats = analyze_trajectory_patterns(rolling_stats)

    return rolling_stats, pca, fig, trajectory_stats


games_df = pd.read_csv('games.csv')
rolling_stats, pca, fig, trajectory_stats = analyze_team_evolution(games_df, window_size=5)
plt.show()
