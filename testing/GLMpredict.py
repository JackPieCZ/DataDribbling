from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.api import GLM, add_constant, families
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, roc_curve


class BasketballPredictor:
    def __init__(self, games_df):
        self.games_df = games_df
        self.team_stats = {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

    def calculate_team_stats(self, window=20):
        """Calculate rolling statistics for each team"""
        stats_columns = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'ORB', 'DRB', 'RB',
                         'AST', 'STL', 'BLK', 'TOV', 'PF', 'SC']

        for team_id in set(self.games_df['HID'].unique()) | set(self.games_df['AID'].unique()):
            # Get all games for the team (both home and away)
            home_games = self.games_df[self.games_df['HID'] == team_id].copy()
            away_games = self.games_df[self.games_df['AID'] == team_id].copy()

            # Create separate dataframes for home and away stats
            home_stats = home_games[['Date'] + [f'H{col}' for col in stats_columns]].copy()
            away_stats = away_games[['Date'] + [f'A{col}' for col in stats_columns]].copy()

            # Rename columns to remove H/A prefix
            home_stats.columns = ['Date'] + stats_columns
            away_stats.columns = ['Date'] + stats_columns

            # Combine home and away stats
            team_stats = pd.concat([home_stats, away_stats]).sort_values('Date')

            # Calculate rolling averages
            rolling_stats = team_stats[stats_columns].rolling(window=window, min_periods=1).mean()
            rolling_stats['Date'] = team_stats['Date']

            self.team_stats[team_id] = rolling_stats

    def prepare_features(self, row):
        """Prepare features for a single game"""
        game_date = row['Date']
        home_id = row['HID']
        away_id = row['AID']

        # Get the most recent stats before the game date
        home_stats = self.team_stats[home_id][self.team_stats[home_id]['Date'] < game_date].iloc[-1]
        away_stats = self.team_stats[away_id][self.team_stats[away_id]['Date'] < game_date].iloc[-1]

        # features = {
        #     'home_fg_pct': home_stats['FGM'] / home_stats['FGA'],
        #     'away_fg_pct': away_stats['FGM'] / away_stats['FGA'],
        #     'home_ft_pct': home_stats['FTM'] / home_stats['FTA'],
        #     'away_ft_pct': away_stats['FTM'] / away_stats['FTA'],
        #     'home_3p_pct': home_stats['FG3M'] / home_stats['FG3A'],
        #     'away_3p_pct': away_stats['FG3M'] / away_stats['FG3A'],
        #     'home_reb_diff': home_stats['RB'] - away_stats['RB'],
        #     'home_ast_diff': home_stats['AST'] - away_stats['AST'],
        #     'home_tov_diff': home_stats['TOV'] - away_stats['TOV'],
        #     'home_scoring': home_stats['SC'],
        #     'away_scoring': away_stats['SC']
        # }
        features = {
            'home_fg_pct': home_stats['FGM'] / home_stats['FGA'] if home_stats['FGA'] != 0 else 0,
            'away_fg_pct': away_stats['FGM'] / away_stats['FGA'] if away_stats['FGA'] != 0 else 0,
            'home_ft_pct': home_stats['FTM'] / home_stats['FTA'] if home_stats['FTA'] != 0 else 0,
            'away_ft_pct': away_stats['FTM'] / away_stats['FTA'] if away_stats['FTA'] != 0 else 0,
            'home_3p_pct': home_stats['FG3M'] / home_stats['FG3A'] if home_stats['FG3A'] != 0 else 0,
            'away_3p_pct': away_stats['FG3M'] / away_stats['FG3A'] if away_stats['FG3A'] != 0 else 0,
            'home_reb_diff': home_stats['RB'] - away_stats['RB'],
            'home_ast_diff': home_stats['AST'] - away_stats['AST'],
            'home_tov_diff': home_stats['TOV'] - away_stats['TOV'],
            'home_scoring': home_stats['SC'],
            'away_scoring': away_stats['SC']
        }


        return pd.Series(features)

    def prepare_training_data(self):
        """Prepare all data for model training"""
        print("Calculating team statistics...")
        self.calculate_team_stats()

        print("Preparing features for each game...")
        features_list = []
        target_list = []

        for _, row in self.games_df.iterrows():
            try:
                features = self.prepare_features(row)
                features_list.append(features)
                target_list.append(row['HSC'] > row['ASC'])
            except (KeyError, IndexError, ZeroDivisionError):
                continue

        X = pd.DataFrame(features_list)
        y = pd.Series(target_list)

        # Store feature columns for later use
        self.feature_columns = X.columns

        return X, y
    
    def evaluate_model(self, X, y, threshold=0.5):
        """Evaluate model performance using various metrics"""
        # Scale features and add constant
        X_scaled = self.scaler.transform(X)
        X_scaled = add_constant(X_scaled)

        # Get predicted probabilities
        y_pred_proba = self.model_results.predict(X_scaled)
        y_pred = (y_pred_proba > threshold).astype(int)

        # Calculate metrics
        auc_roc = roc_auc_score(y, y_pred_proba)
        ppv = precision_score(y, y_pred)  # Positive Predictive Value
        tpr = recall_score(y, y_pred)     # True Positive Rate
        accuracy = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        print("\nModel Evaluation Metrics:")
        print(f"AUC-ROC: {auc_roc:.3f}")
        print(f"Positive Predictive Value: {ppv:.3f}")
        print(f"True Positive Rate: {tpr:.3f}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1 Score: {f1:.3f}")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        return {
            'auc_roc': auc_roc,
            'ppv': ppv,
            'tpr': tpr,
            'accuracy': accuracy,
            'f1': f1
        }

    def train_model(self, test_size=0.2):
        """Train the GLM model"""
        print("Preparing training data...")
        X, y = self.prepare_training_data()

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = add_constant(X_train_scaled)

        print("Training model...")
        self.model = GLM(y_train, X_train_scaled, family=families.Binomial())
        self.model_results = self.model.fit()

        print("\nModel Summary:")
        print(self.model_results.summary())
        print(self.feature_columns)

        # Evaluate model on test set
        print("\nTest Set Evaluation:")
        test_metrics = self.evaluate_model(X_test, y_test)

        # Evaluate model on training set
        print("\nTraining Set Evaluation:")
        train_metrics = self.evaluate_model(X_train, y_train)

        return train_metrics, test_metrics

    def predict_game(self, home_team_id, away_team_id, current_date=None):
        """Predict the outcome of a game between two teams"""
        if current_date is None:
            current_date = datetime.now().strftime('%Y-%m-%d')

        # Create a dummy row with the team IDs and date
        dummy_row = pd.Series({
            'Date': current_date,
            'HID': home_team_id,
            'AID': away_team_id
        })

        """
        features = self.prepare_features(dummy_row)

        # Ensure features are in a DataFrame with the same columns as training data
        features_df = pd.DataFrame([features])
        features_df = features_df[self.feature_columns]  # Reorder columns to match training data

        # Scale features
        features_scaled = pd.DataFrame(
            self.scaler.transform(features_df),
            columns=self.feature_columns
        )
        print(features_scaled)

        # Add constant term
        features_scaled = add_constant(features_scaled, prepend=True)
        print(features_scaled)
        """
        # Prepare features for the matchup
        features = self.prepare_features(dummy_row)

        # Scale features
        features_scaled = self.scaler.transform(features.values.reshape(1, -1))
        print(self.feature_columns)

        # Manually add constant term (1.0) at the start
        features_scaled_with_const = np.insert(features_scaled[0], 0, 1.0)

        # Make prediction
        win_probability = self.model_results.predict(features_scaled_with_const)

        return float(win_probability)


def main():
    # Load the data
    games_df = pd.read_csv(r"D:\_FEL\SAN\project\DataDribbling\data_all\merged_games.csv")
    # games_df = pd.read_csv(r"D:\_FEL\SAN\project\DataDribbling\data0-75-99\games.csv")
    # games_df = pd.read_csv(r"D:\_FEL\SAN\project\DataDribbling\data3-07-09\games.csv")

    # Convert Date column to datetime
    games_df['Date'] = pd.to_datetime(games_df['Date'])

    # Create and train the predictor
    predictor = BasketballPredictor(games_df)
    train_metrics, test_metrics = predictor.train_model()

    # # Example prediction
    # home_team_id = 0
    # away_team_id = 3
    # win_probability = predictor.predict_game(home_team_id, away_team_id)
    # print(f"\nPredicted probability of home team (ID: {
    #       home_team_id}) winning against away team (ID: {away_team_id}): {win_probability:.2%}")


if __name__ == "__main__":
    main()
