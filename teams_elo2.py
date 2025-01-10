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

def reset_elos(team_elo):
    """Reset all teams' ELO ratings to the initial value."""
    for team in team_elo:
        team_elo[team] = INITIAL_ELO
    return team_elo

def calculate_implied_probability(odds):
    """Convert betting odds to implied probability."""
    return 1 / odds

def combine_probabilities(elo_prob, odds_prob, weight_elo=0.65, weight_odds=0.35):
    """Combine probabilities from ELO and betting odds."""
    return weight_elo * elo_prob + weight_odds * odds_prob

def load_initial_elo(filepath):
    """Load initial ELO ratings from a CSV file if it exists."""
    import os
    import pandas as pd

    if os.path.exists(filepath):
        elo_df = pd.read_csv(filepath)
        return dict(zip(elo_df['TeamID'], elo_df['ELO']))
    return {}

def choose_data_directory():
    """Prompt user to select a data directory."""
    import os

    directories = [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('data')]
    print("Available data directories:")
    for i, directory in enumerate(directories):
        print(f"{i + 1}: {directory}")
    
    while True:
        try:
            choice = int(input("Select a data directory by number: "))
            if 1 <= choice <= len(directories):
                return directories[choice - 1]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def main():
    import pandas as pd
    import os

    # Choose data directory
    data_dir = choose_data_directory()
    games_filepath = os.path.join(data_dir, 'games.csv')
    initial_elo_filepath = os.path.join(data_dir, 'teams_elo.csv')

    # Load games data and assign a name to the first column (GameID)
    games = pd.read_csv(games_filepath, header=0)
    
    if games.columns[0] != "GameID":
        games.rename(columns={games.columns[0]: "GameID"}, inplace=True)

    # Load initial ELO ratings if available
    team_elo = load_initial_elo(initial_elo_filepath)

    # Track current season
    current_season = None

    # Statistics for predictions
    total_games = 0
    correct_predictions = 0

    # List to store prediction results
    predictions = []

    reset = input(f"Reset ELO ratings (yes/no): ").strip().lower()
    if reset == "yes":
        team_elo = reset_elos(team_elo)

    # Iterate through each game
    for _, game in games.iterrows():
        # Check if the season has changed
        season = game['Season']
        if current_season is not None and season != current_season:
            # Carry over ELO points to the new season
            team_elo = carry_over_elo(team_elo)
        current_season = season

        # Extract team IDs, scores, and betting odds
        home_team = game['HID']
        away_team = game['AID']
        home_score = game['HSC']
        away_score = game['ASC']
        home_odds = game['OddsH']
        away_odds = game['OddsA']

        # Calculate implied probabilities from odds
        home_odds_prob = calculate_implied_probability(home_odds)
        away_odds_prob = calculate_implied_probability(away_odds)

        # Get current ELO ratings, initializing if necessary
        home_elo = team_elo.get(home_team, INITIAL_ELO)
        away_elo = team_elo.get(away_team, INITIAL_ELO)

        # Calculate expected outcomes from ELO
        home_elo_prob = calculate_expected_outcome(home_elo, away_elo)
        away_elo_prob = calculate_expected_outcome(away_elo, home_elo)

        # Combine probabilities
        home_combined_prob = combine_probabilities(home_elo_prob, home_odds_prob)
        away_combined_prob = combine_probabilities(away_elo_prob, away_odds_prob)

        # Predict winner
        predicted_winner = home_team if home_elo >= away_elo else away_team
        actual_winner = home_team if home_score > away_score else (away_team if away_score > home_score else "Draw")

        # Update prediction statistics
        total_games += 1
        if predicted_winner == actual_winner:
            correct_predictions += 1

        # Store prediction result
        predictions.append({
            "GameID": game["GameID"],
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "PredictedWinner": predicted_winner,
            "ActualWinner": actual_winner,
            "CorrectPrediction": predicted_winner == actual_winner
        })

        # Determine the actual result (1 for win, 0.5 for tie, 0 for loss)
        if home_score > away_score:
            home_result = 1
            away_result = 0
        elif home_score < away_score:
            home_result = 0
            away_result = 1
        else:
            home_result = away_result = 0.5

        # Update ELO ratings
        team_elo[home_team] = update_elo(home_elo, home_elo_prob, home_result)
        team_elo[away_team] = update_elo(away_elo, away_elo_prob, away_result)

    # Save updated ELO ratings to a CSV file
    elo_df = pd.DataFrame(list(team_elo.items()), columns=['TeamID', 'ELO'])
    elo_df.to_csv(initial_elo_filepath, index=False)

    # Save prediction statistics to a CSV file
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(os.path.join(data_dir, 'elo_predictions.csv'), index=False)

    # Calculate and print prediction accuracy
    accuracy = (correct_predictions / total_games) * 100
    print(f"Prediction Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()