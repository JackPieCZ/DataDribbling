import pandas as pd
import numpy as np


def calculate_win_probs_sviga(summary, opp, games_inc, players_inc, database):
    """Calculates win probabilities for home and away team.

        Args:
            summary (pd.Dataframe): Summary of games with columns | Bankroll | Date | Min_bet | Max_bet |.
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].
            games_inc (pd.Dataframe): Incremental data for games.
            players_inc (pd.Dataframe): Incremental data for players.
            database (HistoricalDatabase): Database storing all past incremental data.

        Returns:
            tuple(float, float): Probability of home team winning, probability of away team winning.
        """
    # Example use of opp
    current_season = opp['Season']
    match_date = opp['Date']
    home_ID = opp['HID']
    away_ID = opp['AID']
    neutral_ground = opp['N']
    playoff_game = opp['POFF']
    oddsH = opp['OddsH']
    oddsA = opp['OddsA']

    # Example use of summary
    bankroll = summary['Bankroll']
    current_date = summary['Date']
    min_bet = summary['Min_bet']
    max_bet = summary['Max_bet']

    # Example use of database
    home_team_games_stats = database.get_team_data(home_ID)
    # print(f"Last two games of home team:\n {home_team_games_stats.tail(2)}")
    away_team_game_stats = database.get_team_data(away_ID)

    player3048_stats = database.get_player_data(3048)
    # print(f"Last two games of player 3048:\n {player3048_stats.tail(2)}")
    home_win_prob = 0.5
    away_win_prob = 0.5
    print(f"Calculated win probabilities: {home_win_prob}, {away_win_prob}")
    input("Press Enter to confirm and continue...")
    return home_win_prob, away_win_prob


def calculate_win_probs_sviga2(summary, opp, games_inc, players_inc, database):
    """Calculates win probabilities for home and away team.

        Args:
            summary (pd.Dataframe): Summary of games with columns | Bankroll | Date | Min_bet | Max_bet |.
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].
            games_inc (pd.Dataframe): Incremental data for games.
            players_inc (pd.Dataframe): Incremental data for players.
            database (HistoricalDatabase): Database storing all past incremental data.

        Returns:
            tuple(float, float): Probability of home team winning, probability of away team winning.
        """
    # Example use of opp
    current_season = opp['Season']
    match_date = opp['Date']
    home_ID = opp['HID']
    away_ID = opp['AID']
    neutral_ground = opp['N']
    playoff_game = opp['POFF']
    oddsH = opp['OddsH']
    oddsA = opp['OddsA']

    # Example use of summary
    bankroll = summary['Bankroll']
    current_date = summary['Date']
    min_bet = summary['Min_bet']
    max_bet = summary['Max_bet']

    # Example use of database
    home_team_games_stats = database.get_team_data(home_ID)
    # print(f"Last two games of home team:\n {home_team_games_stats.tail(2)}")
    away_team_game_stats = database.get_team_data(away_ID)

    player3048_stats = database.get_player_data(3048)
    # print(f"Last two games of player 3048:\n {player3048_stats.tail(2)}")
    home_win_prob = 0.5
    away_win_prob = 0.5
    print(f"Calculated win probabilities: {home_win_prob}, {away_win_prob}")
    input("Press Enter to confirm and continue...")
    return home_win_prob, away_win_prob
