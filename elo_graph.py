import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./data_all/tracked_teams_elo.csv')  # Adjust the file path as necessary

# Filter data for the specified teams and season
team_ids = [15, 24, 20, 41]
season = 5
filtered_data = data[(data['TeamID'].isin(team_ids)) & (data['Season'] == season)]

# Sort data by GameID to ensure chronological order
filtered_data = filtered_data.sort_values(by='GameID')

# Plot ELO ratings for each team
plt.figure(figsize=(12, 6))
for team_id in team_ids:
    team_data = filtered_data[filtered_data['TeamID'] == team_id]
    plt.plot(team_data['GameID'], team_data['ELO'], label=f'Team {team_id}')

# Customize the plot
plt.title(f'ELO Rating Season Development')
plt.xlabel('')
plt.ylabel('ELO Rating')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()