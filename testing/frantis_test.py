<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
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
    result = matches[['Date', 'team1_score', 'team2_score']].sort_values(by='Date').reset_index(drop=True)
    
    #compare_score
    """Adds column: team1_score/team2_score"""
    result = result.assign(score_ratio=result['team1_score'] / result['team2_score'])
    return result


# Example usage
data = pd.read_csv('./testing/data/games.csv')

games_df = pd.DataFrame(data)

# Find matches between TeamA and TeamB
<<<<<<< Updated upstream
matches = score_team_matches(games_df, 1, 2)
=======
matches = get_team_matches(games_df, 1, 2)
>>>>>>> Stashed changes
print(matches)


