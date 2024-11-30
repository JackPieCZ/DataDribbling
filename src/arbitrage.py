import pandas as pd


def calculate_arbitrage_betting(odds_a, odds_b):
    # Step 1: Calculate the implied probabilities for each outcome
    prob_a = 1 / odds_a
    prob_b = 1 / odds_b

    # Step 2: Calculate the sum of the implied probabilities
    total_prob = prob_a + prob_b

    # Step 3: Check if there is an arbitrage opportunity (total probability < 1)
    if total_prob < 1:
        return True
    else:
        return False


if __name__ == "__main__":

    for season_number in range(1, 24):
        # Prompt the user to enter the season number
        # season_number = int(input("Enter the season number: "))

        # Load the data from the "data" folder
        games = pd.read_csv('./data/games.csv')  # Adjust the path if necessary

        # Filter for the specified season and regular season games (POFF = 0)
        season_games = games[(games['Season'] == season_number)]
                            #& (games['POFF'] == 0)]

        arbitrage_opp = 0

        # Iterate over the filtered season games
        for _, row in games.iterrows():
            OddsH = row['OddsH']
            OddsA = row['OddsA']

            if calculate_arbitrage_betting(OddsH, OddsA):
                arbitrage_opp += 1

        print(f"Number of arbitrage opportunities: {arbitrage_opp}")
