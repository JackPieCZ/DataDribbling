"""Simplest way to decide winner based on thier score in previos matches you can adjust number of last matches (how_many) which is used and
 threshold when you decide to predict winner"""
import pandas as pd


def score_team_matches(games_df, team1, team2):
    """
    Get the dates and scores of matches between two teams in specified order.
    Then it divides score
    Args:
        games_df (pd.DataFrame): DataFrame containing games data with columns 
                                 ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score'].
        team1 (str): Name or ID of the first team.
        team2 (str): Name or ID of the second team.

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'team1_score', 'team2_score', 'score_ratio'].
    """
    # Filter matches where team1 and team2 played against each other
    matches = games_df[
        ((games_df['HID'] == team1) & (games_df['AID'] == team2)) |
        ((games_df['HID'] == team2) & (games_df['AID'] == team1))
    ]

    # Prepare result with team1 and team2 scores in the correct order
    matches = matches.assign(
        team1_score=matches.apply(
            lambda row: row['HSC'] if row['HID'] == team1 else row['ASC'], axis=1
        ),
        team2_score=matches.apply(
            lambda row: row['ASC'] if row['HID'] == team1 else row['HSC'], axis=1
        )
    )

    # Select and sort the relevant columns
    result = matches[['Date', 'team1_score', 'team2_score']
                     ].sort_values(by='Date').reset_index(drop=True)

    # compare_score
    """Adds column: team1_score/team2_score"""
    result = result.assign(
        score_ratio=result['team1_score'] / result['team2_score'])
    return result


def calculate_winner_score_ratio(matches, how_many=None):
    # Sort by date and filter the last `how_many` matches
    matches = matches.sort_values(by='Date', ascending=False)
    if how_many:
        matches = matches.head(how_many)

    # Calculate cumulative score ratio (average)
    cumulative_score_ratio = matches['score_ratio'].mean()

    # Predict the winner based on cumulative score ratio
    threshold = 0.1
    if cumulative_score_ratio > (1 + threshold):
        predicted_winner = 1  # Team 1 wins
    elif cumulative_score_ratio < (1 - threshold):
        predicted_winner = 2  # Team 2 wins
    else:
        predicted_winner = 0  # No clear winner (close game)

    return predicted_winner


# Example usage
data = pd.read_csv('./data/games.csv')

games_df = pd.DataFrame(data)

# Find matches between TeamA and TeamB
matches = score_team_matches(games_df, 1, 2)
print(matches)
print("winner:", calculate_winner_score_ratio(matches, 20))
