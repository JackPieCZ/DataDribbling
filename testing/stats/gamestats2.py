import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Set the style for all visualizations
plt.style.use('seaborn')
sns.set_palette("husl")


def load_and_prepare_data(file_path):
    """Load and prepare the dataset for analysis."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year

    # Calculate additional metrics
    df['Home_FG_Percentage'] = (df['HFGM'] / df['HFGA'] * 100).round(2)
    df['Away_FG_Percentage'] = (df['AFGM'] / df['AFGA'] * 100).round(2)
    df['Home_3P_Percentage'] = (df['HFG3M'] / df['HFG3A'] * 100).round(2)
    df['Away_3P_Percentage'] = (df['AFG3M'] / df['AFG3A'] * 100).round(2)
    df['Home_FT_Percentage'] = (df['HFTM'] / df['HFTA'] * 100).round(2)
    df['Away_FT_Percentage'] = (df['AFTM'] / df['AFTA'] * 100).round(2)

    return df


def create_scoring_distribution(df):
    """Create a violin plot showing the distribution of scores for home and away teams."""
    plt.figure(figsize=(10, 6))

    # Create violin plot
    scoring_data = pd.DataFrame({
        'Score': pd.concat([df['HSC'], df['ASC']]),
        'Team': ['Home']*len(df) + ['Away']*len(df)
    })

    sns.violinplot(x='Team', y='Score', data=scoring_data)
    plt.title('Distribution of Scores: Home vs Away Teams')
    plt.ylabel('Points Scored')
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_shooting_efficiency_comparison(df):
    """Create a bar plot comparing shooting percentages."""
    plt.figure(figsize=(12, 6))

    # Calculate average percentages
    metrics = {
        'Field Goal %': (df['Home_FG_Percentage'].mean(), df['Away_FG_Percentage'].mean()),
        '3-Point %': (df['Home_3P_Percentage'].mean(), df['Away_3P_Percentage'].mean()),
        'Free Throw %': (df['Home_FT_Percentage'].mean(), df['Away_FT_Percentage'].mean())
    }

    x = range(len(metrics))
    width = 0.35

    plt.bar([i - width/2 for i in x], [m[0] for m in metrics.values()], width, label='Home', color='skyblue')
    plt.bar([i + width/2 for i in x], [m[1] for m in metrics.values()], width, label='Away', color='lightcoral')

    plt.ylabel('Percentage')
    plt.title('Shooting Efficiency Comparison: Home vs Away')
    plt.xticks(x, metrics.keys())
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_statistical_summary(df):
    """Create a statistical summary table."""
    stats = pd.DataFrame({
        'Home Team': {
            'Average Score': df['HSC'].mean().round(2),
            'Field Goal %': df['Home_FG_Percentage'].mean().round(2),
            '3-Point %': df['Home_3P_Percentage'].mean().round(2),
            'Free Throw %': df['Home_FT_Percentage'].mean().round(2),
            'Rebounds': df['HRB'].mean().round(2),
            'Assists': df['HAST'].mean().round(2),
            'Steals': df['HSTL'].mean().round(2),
            'Blocks': df['HBLK'].mean().round(2),
            'Turnovers': df['HTOV'].mean().round(2)
        },
        'Away Team': {
            'Average Score': df['ASC'].mean().round(2),
            'Field Goal %': df['Away_FG_Percentage'].mean().round(2),
            '3-Point %': df['Away_3P_Percentage'].mean().round(2),
            'Free Throw %': df['Away_FT_Percentage'].mean().round(2),
            'Rebounds': df['ARB'].mean().round(2),
            'Assists': df['AAST'].mean().round(2),
            'Steals': df['ASTL'].mean().round(2),
            'Blocks': df['ABLK'].mean().round(2),
            'Turnovers': df['ATOV'].mean().round(2)
        }
    })
    return stats


def create_win_percentage_by_odds(df):
    """Create a scatter plot showing relationship between odds and actual wins."""
    plt.figure(figsize=(10, 6))

    plt.scatter(df['OddsH'], df['H'], alpha=0.5, label='Home Teams')
    plt.scatter(df['OddsA'], df['A'], alpha=0.5, label='Away Teams')

    plt.xlabel('Betting Odds')
    plt.ylabel('Win (1) or Loss (0)')
    plt.title('Relationship Between Betting Odds and Game Outcomes')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_rebounding_comparison(df):
    """Create a box plot comparing offensive and defensive rebounds."""
    plt.figure(figsize=(12, 6))

    # Prepare data for plotting
    rebounds_data = pd.DataFrame({
        'Offensive Rebounds': pd.concat([df['HORB'], df['AORB']]),
        'Defensive Rebounds': pd.concat([df['HDRB'], df['ADRB']]),
        'Team': ['Home']*len(df) + ['Away']*len(df)
    })

    # Melt the DataFrame for easier plotting
    rebounds_melted = pd.melt(rebounds_data, id_vars=['Team'],
                              value_vars=['Offensive Rebounds', 'Defensive Rebounds'],
                              var_name='Rebound Type', value_name='Count')

    # Create box plot
    sns.boxplot(x='Team', y='Count', hue='Rebound Type', data=rebounds_melted)
    plt.title('Rebounding Distribution: Home vs Away Teams')
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def create_game_flow_metrics(df):
    """Create a radar chart comparing various game flow metrics."""
    plt.figure(figsize=(10, 10))

    # Calculate averages for various metrics
    metrics = ['AST', 'STL', 'BLK', 'TOV', 'PF']
    home_stats = [df[f'H{metric}'].mean() for metric in metrics]
    away_stats = [df[f'A{metric}'].mean() for metric in metrics]
    # Calculate combined average for both teams
    # all_stats = [(h + a)/2 for h, a in zip(home_stats, away_stats)]

    # Set up the angles for the radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)

    # Close the plot by appending the first value to the end
    home_stats = np.concatenate((home_stats, [home_stats[0]]))
    away_stats = np.concatenate((away_stats, [away_stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    # Plot
    ax = plt.subplot(111, projection='polar')
    ax.plot(angles, home_stats, 'o-', linewidth=2, label='Home Team')
    ax.fill(angles, home_stats, alpha=0.25)
    ax.plot(angles, away_stats, 'o-', linewidth=2, label='Away Team')
    ax.fill(angles, away_stats, alpha=0.25)
    # ax.plot(angles, all_stats, 'o-', linewidth=2, label='Teams average')
    # ax.fill(angles, away_stats, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Assists', 'Steals', 'Blocks', 'Turnovers', 'Fouls'])
    plt.title('Game Flow Metrics Comparison')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    return plt.gcf()


def create_scoring_trends(df):
    """Create a line plot showing scoring trends over time."""
    plt.figure(figsize=(12, 6))

    # Calculate moving averages for smoother trends
    window_size = 10
    df['Home_MA'] = df['HSC'].rolling(window=window_size).mean()
    df['Away_MA'] = df['ASC'].rolling(window=window_size).mean()

    plt.plot(df.index, np.array(df['Home_MA']), label='Home Teams', color='skyblue')
    plt.plot(df.index, np.array(df['Away_MA']), label='Away Teams', color='lightcoral')

    plt.xlabel('Game Number')
    plt.ylabel('Points Scored (Moving Average)')
    plt.title(f'Scoring Trends ({window_size}-Game Moving Average)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def main():
    # Load and prepare the data
    df = load_and_prepare_data(r"D:\_FEL\SAN\project\DataDribbling\data_all\merged_games.csv")

    # Create visualizations
    scoring_dist = create_scoring_distribution(df)
    shooting_eff = create_shooting_efficiency_comparison(df)
    odds_analysis = create_win_percentage_by_odds(df)
    rebounding_comp = create_rebounding_comparison(df)
    game_flow = create_game_flow_metrics(df)
    # scoring_trends = create_scoring_trends(df)
    stats_summary = create_statistical_summary(df)

    # Save visualizations
    scoring_dist.savefig('scoring_distribution.pdf', bbox_inches='tight', dpi=300)
    shooting_eff.savefig('shooting_efficiency.pdf', bbox_inches='tight', dpi=300)
    odds_analysis.savefig('odds_analysis.pdf', bbox_inches='tight', dpi=300)
    rebounding_comp.savefig('rebounding_comparison.pdf', bbox_inches='tight', dpi=300)
    game_flow.savefig('game_flow_metrics.pdf', bbox_inches='tight', dpi=300)
    # scoring_trends.savefig('scoring_trends.pdf', bbox_inches='tight', dpi=300)

    # Save statistical summary
    stats_summary.to_csv('statistical_summary2.csv')

    # Display basic dataset information
    print("\nDataset Summary:")
    print(f"Total number of games: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"\nHome team win percentage: {(df['H'].mean() * 100).round(2)}%")

    plt.close('all')


if __name__ == "__main__":
    main()
