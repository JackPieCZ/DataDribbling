import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class Model:

    def __init__(self):
        # Initialize the logistic regression model
        self.logreg = LogisticRegression(max_iter=1000)
        self.trained = False
        self.historical_data = pd.DataFrame()

    def train_model(self, games: pd.DataFrame, players: pd.DataFrame):
        # Feature Engineering and Aggregation
        # player_agg = players.groupby(['Season', 'Date', 'Team']).agg({
        #     'MIN': 'sum',
        #     'FGM': 'sum',
        #     'FGA': 'sum',
        #     'FG3M': 'sum',
        #     'FG3A': 'sum',
        #     'FTM': 'sum',
        #     'FTA': 'sum',
        #     'ORB': 'sum',
        #     'DRB': 'sum',
        #     'RB': 'sum',
        #     'AST': 'sum',
        #     'STL': 'sum',
        #     'BLK': 'sum',
        #     'TOV': 'sum',
        #     'PF': 'sum',
        #     'PTS': 'sum'
        # }).reset_index()

        # Merge aggregated player stats back to games data
        # games = games.merge(player_agg, left_on=['Season', 'Date', 'HID'], right_on=[
        #                     'Season', 'Date', 'Team'], suffixes=('', '_H'))
        # games = games.merge(player_agg, left_on=['Season', 'Date', 'AID'], right_on=[
        #                     'Season', 'Date', 'Team'], suffixes=('', '_A'))

        # Drop redundant columns if they exist
        for col in ['Team_H', 'Team_A']:
            if col in games.columns:
                games = games.drop(columns=[col])


        # Feature Selection - Selecting key features for modeling
        features = [
            'HFGM', 'AFGM', 'HFGA', 'AFGA', 'HFG3M', 'AFG3M', 'HFG3A', 'AFG3A',
            'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB', 'HDRB', 'ADRB',
            'HRB', 'ARB', 'HAST', 'AAST', 'HSTL', 'ASTL', 'HBLK', 'ABLK',
            'HTOV', 'ATOV', 'HPF', 'APF'
        ]
        X = games[features]
        # Define target variable (1 if Home team wins, 0 otherwise)
        y = games['H']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0, random_state=42)

        # Train the Logistic Regression Model
        self.logreg.fit(X_train, y_train)
        self.trained = True

    def update_historical_data(self, games: pd.DataFrame, players: pd.DataFrame):
        # Append new games and players data to historical dataset
        self.historical_data = pd.concat([self.historical_data, games], ignore_index=True)

    def calculate_team_features(self, team_id, date):
        # Calculate historical statistics for the given team up to the given date
        team_games = self.historical_data[
            (
                (self.historical_data['HID'] == team_id) | (self.historical_data['AID'] == team_id)
            ) & (self.historical_data['Date'] < date)]
        if team_games.empty:
            return np.zeros(28)  # Return zeros if no historical data is available

        # Calculate averages for relevant features
        features = [
            'HFGM', 'AFGM', 'HFGA', 'AFGA', 'HFG3M', 'AFG3M', 'HFG3A', 'AFG3A',
            'HFTM', 'AFTM', 'HFTA', 'AFTA', 'HORB', 'AORB', 'HDRB', 'ADRB',
            'HRB', 'ARB', 'HAST', 'AAST', 'HSTL', 'ASTL', 'HBLK', 'ABLK',
            'HTOV', 'ATOV', 'HPF', 'APF'
        ]
        team_features = []
        for feature in features:
            home_feature = team_games[team_games['HID'] == team_id][feature].mean()
            away_feature = team_games[team_games['AID'] == team_id][feature].mean()
            team_features.append(np.nanmean([home_feature, away_feature]))
        return team_features

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        # Update historical data if available
        games, players = inc
        if not games.empty:
            self.update_historical_data(games, players)

        # Ensure model is trained if historical data is sufficient
        if not self.trained and len(self.historical_data) > 50:  # Threshold to ensure enough data
            self.train_model(self.historical_data, players)

        # Extract relevant data
        bankroll = summary['Bankroll'].iloc[0]
        min_bet = summary['Min_bet'].iloc[0]
        max_bet = summary['Max_bet'].iloc[0]

        # Prepare feature data for upcoming opportunities
        bets = []
        for idx, row in opps.iterrows():
            # Calculate features for home and away teams based on historical data
            home_features = self.calculate_team_features(row['HID'], row['Date'])
            away_features = self.calculate_team_features(row['AID'], row['Date'])

            # Combine home and away features for prediction
            X_opps = np.array(home_features + away_features).reshape(1, -1)

            # Predict probabilities using the trained model if trained
            if self.trained:
                home_win_prob = self.logreg.predict_proba(X_opps)[:, 1][0]
                away_win_prob = 1 - home_win_prob
            else:
                # If model is not trained, use default 50% probabilities
                home_win_prob = 0.5
                away_win_prob = 0.5

            oddsH = row['OddsH']
            oddsA = row['OddsA']

            # Calculate Kelly bet sizes
            f_H = (home_win_prob * (oddsH - 1) - (1 - home_win_prob)) / (oddsH - 1)
            f_A = (away_win_prob * (oddsA - 1) - (1 - away_win_prob)) / (oddsA - 1)

            # Bet sizes should be between min and max bets and be non-negative
            betH = max(min(f_H * bankroll, max_bet), 0) if f_H > 0 else 0
            betA = max(min(f_A * bankroll, max_bet), 0) if f_A > 0 else 0

            # Append the bets to the DataFrame
            bets.append([row['ID'], betH, betA])

        # Convert list of bets to DataFrame and return
        bets_df = pd.DataFrame(bets, columns=['ID', 'BetH', 'BetA'])
        return bets_df
