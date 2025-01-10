import numpy as np
import pandas as pd

# Sample data
data = pd.read_csv('./data_all/merged_games.csv') 

# Create DataFrame
games = pd.DataFrame(data)

#season_matches = df[(df["Season"] == 1)].copy()

# Add a new column by applying a function
def calculate_home_acc_per(row):
    hgm = row['HFGM']  
    hga = row['HFGA']  
    return round(hgm/hga,4)

def calculate_away_acc_per(row):
    agm = row['AFGM'] 
    aga = row['AFGA']
    return round(agm/aga,4)

games['HA'] = games.apply(calculate_home_acc_per, axis=1)
games['AA'] = games.apply(calculate_away_acc_per, axis=1)

games.drop(
    ['Season', 'Date','Open', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'HFG3M', 'AFG3M', 
     'HFG3A', 'AFG3A', 'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB', 'HDRB', 'ADRB', 
      'HAST', 'AAST', 'HSTL', 'ASTL', 'HBLK', 'ABLK', 'HTOV', 'ATOV', 
     'HPF', 'APF','HFGM','HFGA','AFGM','AFGA'], 
    inplace=True, 
    axis=1, 
    errors='ignore'
)


home_df = games[['H', 'HSC', 'HA','HRB']].rename(columns={'H': 'Team', 'HSC': 'SC', 'HA': 'ACC','HRB': 'RB'})
home_df['Team'] = 'H'  # Mark as home team
home_df['W'] = [1 if row['HSC'] > row['ASC'] else 0 if row['HSC'] < row['ASC'] else 'Draw'
                     for _, row in games.iterrows()]

# Add win/loss information for away team
away_df = games[['A', 'ASC', 'AA','ARB']].rename(columns={'A': 'Team', 'ASC': 'SC', 'AA': 'ACC','ARB': 'RB'})
away_df['Team'] = 'A'  # Mark as away team
away_df['W'] = [1 if row['ASC'] > row['HSC'] else 0 if row['ASC'] < row['HSC'] else 'Draw'
                     for _, row in games.iterrows()]

# Combine the transformed DataFrames without sorting
season_matches = pd.concat([home_df, away_df]).reset_index(drop=True)


# Save to CSV
season_matches.to_csv("./data_all/matches_custom.csv", index=False)