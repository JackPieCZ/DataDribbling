from torch import tensor, nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout=0.4):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU with LeakyReLU
            nn.BatchNorm1d(output_size),
            nn.Dropout(dropout)
        )
        # Shortcut for residual connection
        self.shortcut = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x) + self.shortcut(x)


class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, batch_first=True)
        self.block1 = ResidualBlock(128, 256, dropout=0.5)
        self.block2 = ResidualBlock(256, 128, dropout=0.4)
        self.block3 = ResidualBlock(128, 64, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.Mish(),  # Mish activation function
            nn.BatchNorm1d(32),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x should be of shape (batch_size, input_size)
        # Shape it to (batch_size, seq_len, input_size) -> (1, 1, 320)
        x = x.view(1, 1, -1)
        # Now LSTM input is correctly shaped (batch_size, seq_len, input_size)
        x, _ = self.lstm(x)
        # Removing the seq_len dimension after LSTM output -> Shape: (batch_size, 128)
        x = x.squeeze(1)
        x = self.block1(x)  # Residual Block (output shape: batch_size, 256)
        x = self.block2(x)  # Residual Block (output shape: batch_size, 128)
        x = self.block3(x)  # Residual Block (output shape: batch_size, 64)
        return self.fc(x)   # Fully connected layer to get final output


def calculate_win_probs_kuba(summary, opp, games_inc, players_inc, database):
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
    # print(f"Calculated win probabilities: {home_win_prob}, {away_win_prob}")
    # input("Press Enter to confirm and continue...")
    return home_win_prob, away_win_prob


def calculate_win_probs_kuba2(opp, database, model):
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
    max_sc, max_fgm, max_fga, max_fg3m, max_fg3a, max_ftm, max_fta, max_orb, max_drb, max_rb, max_ast, max_stl, max_blk, max_tov, max_pf = 173, 67, 130, 23, 49, 61, 80, 37, 54, 76, 52, 27, 23, 40, 52
    home_ID = opp['HID']
    away_ID = opp['AID']
    # Example use of database
    home_data = database.get_team_data(home_ID).tail(5)
    away_data = database.get_team_data(away_ID).tail(5)
    home_win_prob = None
    away_win_prob = None

    if len(home_data) == 5 and len(away_data) == 5:
        columns_to_drop = ['GameID', 'Season', 'Date', 'TeamID',
                           'OpponentID', 'TeamOdds', 'OpponentOdds', 'N', 'POFF']
        for data_df in [home_data, away_data]:
            data_df['SC'] /= max_sc
            data_df['OpponentSC'] /= max_sc
            data_df['FGM'] /= max_fgm
            data_df['OpponentFGM'] /= max_fgm
            data_df['FGA'] /= max_fga
            data_df['OpponentFGA'] /= max_fga
            data_df['FG3M'] /= max_fg3m
            data_df['OpponentFG3M'] /= max_fg3m
            data_df['FG3A'] /= max_fg3a
            data_df['OpponentFG3A'] /= max_fg3a
            data_df['FTM'] /= max_ftm
            data_df['OpponentFTM'] /= max_ftm
            data_df['FTA'] /= max_fta
            data_df['OpponentFTA'] /= max_fta
            data_df['ORB'] /= max_orb
            data_df['OpponentORB'] /= max_orb
            data_df['DRB'] /= max_drb
            data_df['OpponentDRB'] /= max_drb
            data_df['RB'] /= max_rb
            data_df['OpponentRB'] /= max_rb
            data_df['AST'] /= max_ast
            data_df['OpponentAST'] /= max_ast
            data_df['STL'] /= max_stl
            data_df['OpponentSTL'] /= max_stl
            data_df['BLK'] /= max_blk
            data_df['OpponentBLK'] /= max_blk
            data_df['TOV'] /= max_tov
            data_df['OpponentTOV'] /= max_tov
            data_df['PF'] /= max_pf
            data_df['OpponentPF'] /= max_pf
        home_team_games_stats = home_data[::-1].drop(
            columns=columns_to_drop, inplace=False).values.tolist()
        away_team_game_stats = away_data[::-1].drop(
            columns=columns_to_drop, inplace=False).values.tolist()

        all_data = []
        for game_data in home_team_games_stats:
            all_data.extend(game_data)
        for game_data in away_team_game_stats:
            all_data.extend(game_data)
        # print(len(all_data))

        # # Assuming `all_data` is already a list with length 320 representing all features for a single game.
        # all_data_tensor = torch.tensor(all_data).float()

        # # Reshape the tensor to match the LSTM's expected input dimensions
        # # Shape: (batch_size, seq_len, input_size), here (1, 1, 320)
        # all_data_tensor = all_data_tensor.view(1, 1, -1)

        # # Forward pass through the model
        # home_win_prob = model(all_data_tensor).item()
        home_win_prob = model(tensor(all_data).float()).item()
        away_win_prob = 1 - home_win_prob
    return home_win_prob, away_win_prob
