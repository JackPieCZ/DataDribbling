import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the style for all visualizations
plt.style.use('seaborn')
sns.set_palette("husl")


def load_and_prepare_data(file_path):
    """Load and prepare the dataset with combined metrics."""
    df = pd.read_csv(file_path)

    # Calculate combined game statistics
    df['Total_Score'] = df['HSC'] + df['ASC']
    df['Total_FGM'] = df['HFGM'] + df['AFGM']
    df['Total_FGA'] = df['HFGA'] + df['AFGA']
    df['Total_3PM'] = df['HFG3M'] + df['AFG3M']
    df['Total_3PA'] = df['HFG3A'] + df['AFG3A']
    df['Total_FTM'] = df['HFTM'] + df['AFTM']
    df['Total_FTA'] = df['HFTA'] + df['AFTA']
    df['Total_ORB'] = df['HORB'] + df['AORB']
    df['Total_DRB'] = df['HDRB'] + df['ADRB']
    df['Total_RB'] = df['HRB'] + df['ARB']
    df['Total_AST'] = df['HAST'] + df['AAST']
    df['Total_STL'] = df['HSTL'] + df['ASTL']
    df['Total_BLK'] = df['HBLK'] + df['ABLK']
    df['Total_TOV'] = df['HTOV'] + df['ATOV']
    df['Total_PF'] = df['HPF'] + df['APF']

    # Calculate percentages
    df['FG_Percentage'] = (df['Total_FGM'] / df['Total_FGA'] * 100).round(2)
    df['ThreeP_Percentage'] = (df['Total_3PM'] / df['Total_3PA'] * 100).round(2)
    df['FT_Percentage'] = (df['Total_FTM'] / df['Total_FTA'] * 100).round(2)

    return df


def create_game_stats_summary(df):
    """Create a summary of overall game statistics."""
    stats_summary = pd.DataFrame({
        'Metric': [
            'Points per Game',
            'Field Goal %',
            'Three Point %',
            'Free Throw %',
            'Rebounds per Game',
            'Assists per Game',
            'Steals per Game',
            'Blocks per Game',
            'Turnovers per Game',
            'Personal Fouls per Game'
        ],
        'Average': [
            df['Total_Score'].mean(),
            df['FG_Percentage'].mean(),
            df['ThreeP_Percentage'].mean(),
            df['FT_Percentage'].mean(),
            df['Total_RB'].mean(),
            df['Total_AST'].mean(),
            df['Total_STL'].mean(),
            df['Total_BLK'].mean(),
            df['Total_TOV'].mean(),
            df['Total_PF'].mean()
        ],
        'Min': [
            df['Total_Score'].min(),
            df['FG_Percentage'].min(),
            df['ThreeP_Percentage'].min(),
            df['FT_Percentage'].min(),
            df['Total_RB'].min(),
            df['Total_AST'].min(),
            df['Total_STL'].min(),
            df['Total_BLK'].min(),
            df['Total_TOV'].min(),
            df['Total_PF'].min()
        ],
        'Max': [
            df['Total_Score'].max(),
            df['FG_Percentage'].max(),
            df['ThreeP_Percentage'].max(),
            df['FT_Percentage'].max(),
            df['Total_RB'].max(),
            df['Total_AST'].max(),
            df['Total_STL'].max(),
            df['Total_BLK'].max(),
            df['Total_TOV'].max(),
            df['Total_PF'].max()
        ]
    })

    return stats_summary.round(2)


def create_scoring_distribution(df):
    """Create a histogram of total game scores."""
    plt.figure(figsize=(12, 6))

    sns.histplot(data=df, x='Total_Score', bins=30, kde=True)
    plt.title('Distribution of Total Game Scores')
    plt.xlabel('Total Points Scored')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_shooting_percentages(df):
    """Create a box plot of different shooting percentages."""
    plt.figure(figsize=(10, 6))

    # Prepare data for plotting
    data_melted = pd.DataFrame({
        'Percentage Type': ['Field Goal %'] * len(df) + ['Three Point %'] * len(df) + ['Free Throw %'] * len(df),
        'Value': list(df['FG_Percentage']) + list(df['ThreeP_Percentage']) + list(df['FT_Percentage'])
    })

    # Create box plot using the melted data
    sns.boxplot(x='Percentage Type', y='Value', data=data_melted)
    plt.title('Distribution of Shooting Percentages')
    plt.ylabel('Percentage')
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_game_metrics_distribution(df):
    """Create a violin plot of various game metrics."""
    plt.figure(figsize=(14, 7))

    # Prepare data for plotting
    metrics_data = pd.DataFrame({
        'Metric': ['Rebounds'] * len(df) + ['Assists'] * len(df) + ['Steals'] * len(df) +
        ['Blocks'] * len(df) + ['Turnovers'] * len(df),
        'Value': list(df['Total_RB']) + list(df['Total_AST']) + list(df['Total_STL']) +
        list(df['Total_BLK']) + list(df['Total_TOV'])
    })

    # Create violin plot
    sns.violinplot(x='Metric', y='Value', data=metrics_data)
    plt.title('Distribution of Game Metrics')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_correlation_heatmap(df):
    """Create a correlation heatmap of game metrics."""
    plt.figure(figsize=(12, 10))

    # Select relevant columns for correlation
    columns = ['Total_Score', 'Total_RB', 'Total_AST', 'Total_STL',
               'Total_BLK', 'Total_TOV', 'FG_Percentage', 'ThreeP_Percentage']
    correlation = df[columns].corr()

    # Create heatmap
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation between Game Metrics')
    plt.tight_layout()

    return plt.gcf()


def main():
    # Load and prepare the data
    df = load_and_prepare_data(r"D:\_FEL\SAN\project\DataDribbling\data_all\merged_games.csv")

    # Create statistical summary
    # Create statistical summary
    stats_summary = create_game_stats_summary(df)

    # Create visualizations
    # scoring_dist = create_scoring_distribution(df)
    shooting_pct = create_shooting_percentages(df)
    game_metrics = create_game_metrics_distribution(df)
    corr_heatmap = create_correlation_heatmap(df)

    # Save outputs
    stats_summary.to_csv('game_statistics_summary2.csv')
    # scoring_dist.savefig('total_scoring_distribution2.png', bbox_inches='tight', dpi=300)
    shooting_pct.savefig('shooting_percentages2.pdf', bbox_inches='tight', dpi=300)
    game_metrics.savefig('game_metrics_distribution2.pdf', bbox_inches='tight', dpi=300)
    corr_heatmap.savefig('correlation_heatmap2.pdf', bbox_inches='tight', dpi=300)

    # Print summary statistics
    print("\nBasketball Games Summary Statistics:")
    print(stats_summary)

    plt.close('all')


if __name__ == "__main__":
    main()
