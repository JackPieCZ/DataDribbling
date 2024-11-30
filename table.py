import pandas as pd

# Prompt the user to enter the season number
season_number = int(input("Enter the season number: "))

# Load the data from the "data" folder
games = pd.read_csv('./data/games.csv')  # Adjust the path if necessary

# Filter for the specified season and regular season games (POFF = 0)
season_games = games[(games['Season'] == season_number) & (games['POFF'] == 0)]

# Initialize a dictionary to keep track of wins, losses, scores, and home/away games for each team
standings = {}

# Iterate over each game in the season's regular season
for _, row in season_games.iterrows():
    hid = row['HID']  # Home team ID
    aid = row['AID']  # Away team ID
    hsc = row['HSC']  # Home team score
    asc = row['ASC']  # Away team score
    home_win = row['H'] == 1  # Home win switch
    away_win = row['A'] == 1  # Away win switch

    # Initialize teams in standings if they don't exist
    if hid not in standings:
        standings[hid] = {'Wins': 0, 'Losses': 0, 'TotalScore': 0, 'GamesPlayed': 0,
                          'HomeWins': 0, 'HomeGames': 0, 'AwayWins': 0, 'AwayGames': 0}
    if aid not in standings:
        standings[aid] = {'Wins': 0, 'Losses': 0, 'TotalScore': 0, 'GamesPlayed': 0,
                          'HomeWins': 0, 'HomeGames': 0, 'AwayWins': 0, 'AwayGames': 0}

    # Update wins and losses based on game result
    if home_win:
        standings[hid]['Wins'] += 1
        standings[hid]['HomeWins'] += 1
        standings[aid]['Losses'] += 1
    elif away_win:
        standings[aid]['Wins'] += 1
        standings[aid]['AwayWins'] += 1
        standings[hid]['Losses'] += 1

    # Add to the games played counters
    standings[hid]['GamesPlayed'] += 1
    standings[aid]['GamesPlayed'] += 1
    standings[hid]['HomeGames'] += 1
    standings[aid]['AwayGames'] += 1

    # Add the scores to each team's total score
    standings[hid]['TotalScore'] += hsc
    standings[aid]['TotalScore'] += asc

# Convert the standings dictionary to a DataFrame for better readability
standings_df = pd.DataFrame.from_dict(standings, orient='index').reset_index()
standings_df.columns = ['TeamID', 'Wins', 'Losses', 'TotalScore', 'GamesPlayed',
                        'HomeWins', 'HomeGames', 'AwayWins', 'AwayGames']

# Calculate average score and win percentages
standings_df['AvgScore'] = standings_df['TotalScore'] / standings_df['GamesPlayed']
standings_df['HomeWinPct'] = (standings_df['HomeWins'] / standings_df['HomeGames']).round(3)
standings_df['AwayWinPct'] = (standings_df['AwayWins'] / standings_df['AwayGames']).round(3)

# Sort standings by Wins in descending order and reset index starting from 1
standings_df = standings_df.sort_values(by='Wins', ascending=False).reset_index(drop=True)
standings_df.index += 1  # Start the index from 1 instead of 0

# Display the season number at the top of the standings
print(f"\nSeason {season_number} Standings")
print(standings_df[['TeamID', 'Wins', 'Losses', 'AvgScore', 'HomeWinPct', 'AwayWinPct']])
