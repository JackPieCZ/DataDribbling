import numpy as np  # noqa
import pandas as pd

from arbitrage import calculate_arbitrage_betting
from database import HistoricalDatabase


class Model:
    def __init__(self, model=None):
        self.db = HistoricalDatabase()
        self.yesterdays_games = None
        self.yesterdays_bets = None
        self.money_spent_yesterday = 0
        self.bankroll_after_bets = 0
        self.model = model
        if model is not None:
            self.model.eval()
            print("Model loaded")

    def kelly_criterion(self, probability, odds):
        """
        Vypočítá optimální výši sázky pomocí Kellyho kritéria.

        :param probability: odhadovaná pravděpodobnost výhry (0 až 1).
        :param odds: kurz
        # :param bankroll: dostupný kapitál
        # :param fraction: frakční Kelly (např. 0.5 pro poloviční Kelly).
        :return: doporučený zlomek sázky.
        """
        q = 1 - probability
        b = odds - 1  # zisk

        optimal_fraction = probability - (q / b)
        # nesázet, pokud je Kelly záporný
        optimal_fraction = max(0, optimal_fraction)
        return float(optimal_fraction)

    def evaluate_yestedays_predictions(self, bankroll):
        if self.yesterdays_bets is not None:
            # Calculate accuracy of yesterday's predictions
            correct_predictions = 0
            correct_bets = 0
            correct_bookmaker_bets = 0
            total_predictions = len(self.yesterdays_games)
            num_bets = ((self.yesterdays_bets["newBetH"] > 0) | (self.yesterdays_bets["newBetA"] > 0)).sum()

            for idx, game in self.yesterdays_games.iterrows():
                # Get corresponding prediction
                prediction = self.yesterdays_bets.loc[idx]
                # Export prediction with index to CSV
                # prediction_df = pd.DataFrame(prediction).T
                # game_df = pd.DataFrame(game).T

                # Determine which team was predicted to win
                if prediction["ProbH"] == 0 and prediction["ProbA"] == 0:
                    continue
                assert prediction["ProbH"] + prediction["ProbA"] != 0, "Probabilities should not sum up to zero"
                predicted_home_win = prediction["ProbH"] > prediction["ProbA"]

                if prediction["newBetH"] + prediction["newBetA"] != 0:
                    betted_home_win = prediction["newBetH"] > prediction["newBetA"]
                    if (betted_home_win and game["H"] == 1) or (not betted_home_win and game["A"] == 1):
                        correct_bets += 1

                bookmaker_predicted_home_win = game["OddsH"] < game["OddsA"]
                if (bookmaker_predicted_home_win and game["H"] == 1) or (not bookmaker_predicted_home_win and game["A"] == 1):
                    correct_bookmaker_bets += 1

                # Check if prediction matches actual result
                if (predicted_home_win and game["H"] == 1) or (not predicted_home_win and game["A"] == 1):
                    correct_predictions += 1
            pred_accuracy = correct_predictions / total_predictions if total_predictions > 0 else None
            bets_accuracy = correct_bets / num_bets if num_bets > 0 else None
            bookmaker_accuracy = correct_bookmaker_bets / total_predictions if total_predictions > 0 else None
            print(f"Yesterday's prediction accuracy: {pred_accuracy} ({correct_predictions}/{total_predictions})")
            # print(f"Yesterday's betting accuracy: {bets_accuracy} ({correct_bets}/{num_bets})")
            print(
                f"Yesterday's bookmaker's accuracy: {bookmaker_accuracy} ({correct_bookmaker_bets}/{total_predictions})")
            print(
                f"Money - spent: {self.money_spent_yesterday:.2f}$, gained: {bankroll - self.bankroll_after_bets:.2f}$")
            # input("Press Enter to continue...")

    def calculate_kelly(self, opps, todays_budget, kelly_fraction, min_bet, max_bet):
        # Sort Kelly criterion in descending order and keep track of original indices
        # Create a new column with the maximum kelly of home and away
        opps["MaxKelly"] = opps[["KellyH", "KellyA"]].max(axis=1)
        sorted_win_probs_opps = opps.sort_values(by="MaxKelly", ascending=False)

        # Place bets based on Kelly criterion starting with the highest one
        for opp_idx, row in sorted_win_probs_opps.iterrows():
            # opp_idx = row["index"]
            kellyH = row["KellyH"]
            kellyA = row["KellyA"]

            # # New logic: Only bet on the outcome with the higher probability
            # probH = row["ProbH"]
            # probA = row["ProbA"]
            # if probH >= probA:
            #     kellyA = 0  # Set the Kelly fraction for away team to zero if home is predicted higher
            # else:
            #     kellyH = 0  # Set the Kelly fraction for home team to zero if away is predicted higher

            # Skip if both Kelly fractions are zero
            if kellyH == 0 and kellyA == 0:
                continue

            bet_home = kellyH * todays_budget * kelly_fraction
            bet_away = kellyA * todays_budget * kelly_fraction

            # Bet sizes should be between min and max bets and be non-negative
            betH = max(min(bet_home, max_bet), min_bet) if bet_home >= min_bet else 0
            betA = max(min(bet_away, max_bet), min_bet) if bet_away >= min_bet else 0

            # Update the bets DataFrame with calculated bet sizes
            opps.loc[opps.index == opp_idx, "newBetH"] = betH
            opps.loc[opps.index == opp_idx, "newBetA"] = betA
            todays_budget -= betH + betA

            # Stop if we run out of budget
            if todays_budget <= 0:
                break

    def bet_on_higher_odds(self, opps, todays_budget, min_bet, max_bet, non_kelly_bet_amount):
        # Bet on a team we predicted to win
        for opp_idx, row in opps.iterrows():
            probH = row["ProbH"]
            probA = row["ProbA"]
            if probH == 0 and probA == 0:
                continue

            betH = non_kelly_bet_amount if probH >= probA else 0
            betA = non_kelly_bet_amount if probA > probH else 0

            betH = max(min(betH, max_bet), min_bet) if betH >= min_bet else 0
            betA = max(min(betA, max_bet), min_bet) if betA >= min_bet else 0

            opps.loc[opps.index == opp_idx, "newBetH"] = betH
            opps.loc[opps.index == opp_idx, "newBetA"] = betA
            todays_budget -= betH + betA

            if todays_budget <= 0:
                break

    def place_bets(self, summary: pd.DataFrame, opps: pd.DataFrame, inc: tuple[pd.DataFrame, pd.DataFrame]):
        games_inc, players_inc = inc
        bankroll = summary.iloc[0]["Bankroll"]
        min_bet = summary.iloc[0]["Min_bet"]
        if opps.empty and games_inc.empty and players_inc.empty or bankroll < min_bet:
            return pd.DataFrame(columns=["BetH", "BetA"])

        max_bet = summary.iloc[0]["Max_bet"]
        todays_date = summary.iloc[0]["Date"]

        # only iterate over opps with the current date while keeping the original index
        assert opps[opps["Date"] < todays_date].empty, "There are opps before today's date, which should never happen"
        todays_opps = opps[opps["Date"] == todays_date]

        self.yesterdays_games = games_inc

        calculate_win_probs_fn = calculate_win_probs
        kelly_fraction = 0.5
        # fraction of budget we are willing to spend today
        budget_fraction = 0.1
        use_kelly = False
        todays_budget = bankroll * budget_fraction
        non_kelly_bet_amount = min_bet * 2

        # Evaluate yesterday's predictions
        self.evaluate_yestedays_predictions(bankroll)

        self.db.add_incremental_data(games_inc, players_inc)

        # Temporarily disable SettingWithCopyWarning
        pd.options.mode.chained_assignment = None
        # Add columns for new bet sizes and win probabilities
        opps["newBetH"] = 0.0
        opps["newBetA"] = 0.0
        opps["ProbH"] = 0.0
        opps["ProbA"] = 0.0
        opps["KellyH"] = 0.0
        opps["KellyA"] = 0.0

        # Calculate win probabilities for each opportunity
        for opp_idx, opp in todays_opps.iterrows():
            betH = opp["BetH"]
            betA = opp["BetA"]
            oddsH = opp["OddsH"]
            oddsA = opp["OddsA"]
            assert betH == 0 and betA == 0, "Both bets should be zero at the beginning"

            prob_home, prob_away = calculate_win_probs_fn(opp, self.db, self.model)
            if prob_home is None or prob_away is None:
                # print(f"Could not calculate win probabilities for opp {opp_idx}, skipping")
                continue
            assert isinstance(prob_home, (int, float)) and isinstance(
                prob_away, (int, float)
            ), f"Win probabilities should be numbers, currently they are of type {type(prob_home)} and {type(prob_away)}"
            assert 0 <= prob_home <= 1 and 0 <= prob_away <= 1, f"Probabilities should be between 0 and 1, currently they are {prob_home} and {prob_away}"
            assert abs(1 - (prob_home + prob_away)
                       ) < 1e-9, f"Probabilities should sum up to 1, currently they sum up to {prob_home + prob_away}"

            opps.loc[opps.index == opp_idx, "ProbH"] = prob_home
            opps.loc[opps.index == opp_idx, "ProbA"] = prob_away

            # Check if there is an arbitrage betting opportunity
            if calculate_arbitrage_betting(oddsH, oddsA):
                print(f"Arbitrage opportunity detected for opp {opp_idx}, nice!")
                # Take advantage of the arbitrage
                kellyH = 0.5
                kellyA = 0.5
            else:
                # Calculate Kelly bet sizes
                kellyH = self.kelly_criterion(prob_home, oddsH)
                kellyA = self.kelly_criterion(prob_away, oddsA)
                assert kellyH == 0 or kellyA == 0, "Only one kelly should be nonzero, if there is no opportunity to arbitrage"

            opps.loc[opps.index == opp_idx, "KellyH"] = kellyH
            opps.loc[opps.index == opp_idx, "KellyA"] = kellyA

        if use_kelly:
            self.calculate_kelly(opps, todays_budget, kelly_fraction, min_bet, max_bet)
        else:
            self.bet_on_higher_odds(opps, todays_budget, min_bet, max_bet, non_kelly_bet_amount)

        # Do not bet yet
        # self.money_spent_yesterday = bankroll * budget_fraction - todays_budget
        # bets = opps[["newBetH", "newBetA"]]
        # bets.rename(columns={"newBetH": "BetH", "newBetA": "BetA"}, inplace=True)
        # self.yesterdays_bets = opps
        # self.bankroll_after_bets = bankroll - self.money_spent_yesterday
        # return bets
        return pd.DataFrame(columns=["BetH", "BetA"])


def calculate_win_probs(opp, database, model):
    """Calculates win probabilities for home and away team.

        Args:
            opp (pd.Dataframe): Betting opportunities with columns ['Season', 'Date', 'HID', 'AID', 'N', 'POFF', 'OddsH', 'OddsA', 'BetH', 'BetA'].
            database (HistoricalDatabase): Database storing all past incremental data.
            model (Model): Model used for predicting win probabilities.

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

    # # Example use of summary
    # bankroll = summary['Bankroll']
    # current_date = summary['Date']
    # min_bet = summary['Min_bet']
    # max_bet = summary['Max_bet']

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


"""
conda create -n qqh python=3.12.4 -y
conda activate qqh
conda install -c conda-forge -c pytorch -c pyg numpy pandas py-xgboost-cpu scikit-learn scipy statsmodels pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 cpuonly pyg -y
pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
"""
