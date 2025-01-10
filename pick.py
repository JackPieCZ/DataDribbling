import pandas as pd

# Load the data
data = pd.read_csv('./data_all/elo_predictions.csv')

# Filter rows where odds prediction was correct but ELO prediction was incorrect
filtered_data = data[(data['CorrectOddsPrediction'] == False) & (data['CorrectELOPrediction'] == True)]

home_team_wins = filtered_data[filtered_data['HomeTeam'] == filtered_data['ActualWinner']].shape[0]
total_games = filtered_data.shape[0]

# Avoid division by zero
if total_games > 0:
    home_team_win_percentage = (home_team_wins / total_games) * 100
else:
    home_team_win_percentage = 0

# Save the filtered data to a new CSV file
filtered_data.to_csv('./data_all/filtered_predictions4.csv', index=False)


print(f"Percentage of games won by the home team: {home_team_win_percentage:.2f}%")