import pandas as pd

# Define constants
INITIAL_ELO = 1500
K_FACTOR = 20

def calculate_expected_outcome(rating_a, rating_b):
    """Calculate the expected outcome for Team A."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(rating, expected, actual):
    """Update ELO rating based on the result."""
    return rating + K_FACTOR * (actual - expected)

def carry_over_elo(team_elo):
    """Carry over ELO points between seasons."""
    elo_average = sum(team_elo.values()) / len(team_elo)
    for team in team_elo:
        team_elo[team] = (3/4) * team_elo[team] + (1/4) * elo_average
    return team_elo

def main():
 # Load games data
    games = pd.read_csv('./data0-75-99/games.csv')

    # Initialize a dictionary to store team ELO ratings
    team_elo = {}

    # Track current season
    current_season = None

    # Iterate through each game
    for _, game in games.iterrows():
        # Check if the season has changed
        season = game['Season']
        if current_season is not None and season != current_season:
            # Carry over ELO points to the new season
            team_elo = carry_over_elo(team_elo)
        current_season = season

        # Extract team IDs and scores
        home_team = game['HID']
        away_team = game['AID']
        home_score = game['HSC']
        away_score = game['ASC']

        # Determine the actual result (1 for win, 0.5 for tie, 0 for loss)
        if home_score > away_score:
            home_result = 1
            away_result = 0
        elif home_score < away_score:
            home_result = 0
            away_result = 1
        else:
            home_result = away_result = 0.5

        # Get current ELO ratings, initializing if necessary
        home_elo = team_elo.get(home_team, INITIAL_ELO)
        away_elo = team_elo.get(away_team, INITIAL_ELO)

        # Calculate expected outcomes
        home_expected = calculate_expected_outcome(home_elo, away_elo)
        away_expected = calculate_expected_outcome(away_elo, home_elo)

        # Update ELO ratings
        team_elo[home_team] = update_elo(home_elo, home_expected, home_result)
        team_elo[away_team] = update_elo(away_elo, away_expected, away_result)

    # Save updated ELO ratings to a CSV file
    elo_df = pd.DataFrame(list(team_elo.items()), columns=['TeamID', 'ELO'])
    elo_df.to_csv('./data0-75-99/teams_elo.csv', index=False)

if __name__ == '__main__':
    main()
