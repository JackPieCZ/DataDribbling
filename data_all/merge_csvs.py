import pandas as pd

# List of CSV file names
csv_files = [r"D:\_FEL\SAN\project\DataDribbling\data0-75-99\players.csv",
             r"D:\_FEL\SAN\project\DataDribbling\data1-99-05\players.csv",
             r"D:\_FEL\SAN\project\DataDribbling\data2-05-07\players.csv",
             r"D:\_FEL\SAN\project\DataDribbling\data3-07-09\players.csv"]

# Read and concatenate all CSV files
dataframes = []
last_index = -1  # Initialize last index
last_season = 0  # Initialize last season

for file in csv_files:
    df = pd.read_csv(file, float_precision='round_trip')
    df = df.drop(columns=['Unnamed: 0'])

    # Update index to ensure continuity
    df.index = df.index + last_index + 1

    # Update season to ensure continuity
    df['Season'] += last_season

    # Append to list of dataframes
    dataframes.append(df)

    # Update last_index and last_season for the next file
    last_index = df.index[-1]
    last_season = df['Season'].max()

# Combine all dataframes into one
merged_df = pd.concat(dataframes, ignore_index=False)

# Save the merged dataframe to a new CSV
merged_df.to_csv("./merged_players.csv", index=True)

print("CSV files merged successfully into 'merged_players.csv'")
