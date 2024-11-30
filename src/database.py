from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque


class HistoricalDatabase:
    def __init__(self):
        # fmt: off
        self.team_columns = [
            "GameID", "Season", "Date", "TeamID", "OpponentID", "N", "POFF", "TeamOdds", "OpponentOdds", "W", "Home", "SC",
            "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF",
            "OpponentSC", "OpponentFGM", "OpponentFGA", "OpponentFG3M", "OpponentFG3A", "OpponentFTM", "OpponentFTA",
            "OpponentORB", "OpponentDRB", "OpponentRB", "OpponentAST", "OpponentSTL", "OpponentBLK", "OpponentTOV", "OpponentPF"
        ]
        self.player_columns = [
            "PlayerID", "Season", "Date", "TeamID", "GameID", "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "ORB", "DRB", "RB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
        ]
        # fmt: on
        self.team_data: Dict[str, List[Dict]] = defaultdict(list)
        self.player_data: Dict[str, List[Dict]] = defaultdict(list)

    def _create_team_records(self, games_df: pd.DataFrame) -> List[Tuple[str, Dict]]:
        """Create team records maintaining the original game order."""
        records = []

        n_games = len(games_df) * 2
        game_ids = np.repeat(games_df.index.values, 2)
        seasons = np.repeat(games_df["Season"].values, 2)
        dates = np.repeat(games_df["Date"].values, 2)
        n_values = np.repeat(games_df["N"].values, 2)
        poff_values = np.repeat(games_df["POFF"].values, 2)

        team_ids = np.empty(n_games, dtype=object)
        team_ids[0::2] = games_df["HID"].values
        team_ids[1::2] = games_df["AID"].values

        opponent_ids = np.empty(n_games, dtype=object)
        opponent_ids[0::2] = games_df["AID"].values
        opponent_ids[1::2] = games_df["HID"].values

        home_flags = np.empty(n_games, dtype=int)
        home_flags[0::2] = 1
        home_flags[1::2] = 0

        wins = np.empty(n_games, dtype=int)
        wins[0::2] = games_df["H"].values
        wins[1::2] = games_df["A"].values

        team_odds = np.empty(n_games)
        opponent_odds = np.empty(n_games)
        team_odds[0::2] = games_df["OddsH"].values
        team_odds[1::2] = games_df["OddsA"].values
        opponent_odds[0::2] = games_df["OddsA"].values
        opponent_odds[1::2] = games_df["OddsH"].values

        stats_mapping = {
            "SC": ("HSC", "ASC"),
            "FGM": ("HFGM", "AFGM"),
            "FGA": ("HFGA", "AFGA"),
            "FG3M": ("HFG3M", "AFG3M"),
            "FG3A": ("HFG3A", "AFG3A"),
            "FTM": ("HFTM", "AFTM"),
            "FTA": ("HFTA", "AFTA"),
            "ORB": ("HORB", "AORB"),
            "DRB": ("HDRB", "ADRB"),
            "RB": ("HRB", "ARB"),
            "AST": ("HAST", "AAST"),
            "STL": ("HSTL", "ASTL"),
            "BLK": ("HBLK", "ABLK"),
            "TOV": ("HTOV", "ATOV"),
            "PF": ("HPF", "APF"),
        }

        stats_arrays = {}
        for stat, (h_col, a_col) in stats_mapping.items():
            stat_values = np.empty(n_games)
            stat_values[0::2] = games_df[h_col].values
            stat_values[1::2] = games_df[a_col].values
            stats_arrays[stat] = stat_values

        for i in range(n_games):
            record = {
                "GameID": game_ids[i],
                "Season": seasons[i],
                "Date": dates[i],
                "TeamID": team_ids[i],
                "OpponentID": opponent_ids[i],
                "N": n_values[i],
                "POFF": poff_values[i],
                "TeamOdds": team_odds[i],
                "OpponentOdds": opponent_odds[i],
                "Home": home_flags[i],
                "W": wins[i],
                **{stat: stats_arrays[stat][i] for stat in stats_mapping.keys()}
            }

            # Adding opponent stats to the record
            opponent_stats = {
                f"Opponent{stat}": stats_arrays[stat][i ^ 1]  # XOR with 1 to get the opponent's value
                for stat in stats_mapping.keys()
            }

            record.update(opponent_stats)
            records.append((team_ids[i], record))

        return records

    def add_incremental_data(self, games_df: pd.DataFrame, players_df: pd.DataFrame) -> None:
        """Add new game and player data efficiently while maintaining game order."""
        team_records = self._create_team_records(games_df)

        for team_id, record in team_records:
            self.team_data[team_id].append(record)

        # for _, player_games in players_df.groupby("Player"):
        #     player_stats = player_games.rename(columns={"Player": "PlayerID", "Team": "TeamID", "Game": "GameID"})
        #     player_stats = player_stats[self.player_columns]
        #     for _, row in player_stats.iterrows():
        #         self.player_data[row["PlayerID"]].append(row.to_dict())

    def get_team_data(self, team_id: str) -> pd.DataFrame:
        """Get team data, returning empty DataFrame if team not found."""
        return pd.DataFrame(self.team_data.get(team_id, []), columns=self.team_columns)

    def get_player_data(self, player_id: str) -> pd.DataFrame:
        """Get player data, returning empty DataFrame if player not found."""
        return pd.DataFrame(self.player_data.get(player_id, []), columns=self.player_columns)

    def _verify_data_integrity(self) -> None:
        """Verify data integrity using vectorized operations."""
        for team_id, team_records in self.team_data.items():
            df = pd.DataFrame(team_records)
            if not (df["TeamID"] == team_id).all():
                raise ValueError(f"TeamID mismatch in team_data for team {team_id}")
            if not (df["Date"].is_monotonic_increasing or len(df) == 0):
                raise ValueError(f"Games not in chronological order for team {team_id}")

        for player_id, player_records in self.player_data.items():
            df = pd.DataFrame(player_records)
            if not (df["PlayerID"] == player_id).all():
                raise ValueError(f"PlayerID mismatch in player_data for player {player_id}")
