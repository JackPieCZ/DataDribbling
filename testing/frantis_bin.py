import numpy as np
import pandas as pd

# Sample data
data = pd.read_csv('./data/games.csv') 

# Create DataFrame
df = pd.DataFrame(data)

# Filter matches involving team 22
team_22_matches = df[(df["HID"] == 22) | (df["AID"] == 22)].copy()

# Add a column indicating whether team 22 won
team_22_matches["Team22_Won"] = ((team_22_matches["HID"] == 22) & (team_22_matches["H"] == 1)) | \
                                ((team_22_matches["AID"] == 22) & (team_22_matches["A"] == 1))

# Convert the boolean column to integer (0 or 1)
team_22_matches["Team22_Won"] = team_22_matches["Team22_Won"].astype(int)

# Save to CSV
team_22_matches.to_csv("./data/team_22_matches.csv", index=False)

print(team_22_matches)
""""
class Model:
    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        print(f"{summary=}")
        print(f"{opps=}")
        print(f"{inc=}")
        input("Press Enter to continue...")
        min_bet = summary.iloc[0]["Min_bet"]
        N = len(opps)
        bets = np.zeros((N, 2))
        bets[np.arange(N), np.random.choice([0, 1])] = min_bet
        bets = pd.DataFrame(data=bets, columns=["BetH", "BetA"], index=opps.index)
        return bets
"""