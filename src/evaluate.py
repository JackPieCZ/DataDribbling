from  pathlib import Path
import pandas as pd
from model import Model
from environment import Environment

data_folder = "../data0-75-99"

games = pd.read_csv(Path(data_folder) / "games.csv", index_col=0)
games["Date"] = pd.to_datetime(games["Date"])
games["Open"] = pd.to_datetime(games["Open"])

players = pd.read_csv(Path(data_folder) / "players.csv", index_col=0)
players["Date"] = pd.to_datetime(players["Date"])

season_starts = {
    1: "1975-11-07",
    2: "1976-11-12",
    3: "1977-11-11",
    4: "1978-11-10",
    5: "1979-11-09",
    6: "1980-11-07",
    7: "1981-11-13",
    8: "1982-11-12",
    9: "1983-11-11",
    10: "1984-11-09",
    11: "1985-11-08",
    12: "1986-11-07",
    13: "1988-02-12",
    14: "1988-11-08",
    15: "1989-11-07",
    16: "1990-11-06",
    17: "1991-11-05",
    18: "1992-11-03",
    19: "1993-11-09",
    20: "1994-11-08",
    21: "1995-11-07",
    22: "1996-11-05",
    23: "1997-11-04",
    24: "1998-11-03",
}

env = Environment(
    games, players, Model(), init_bankroll=1000, min_bet=5, max_bet=100,
    start_date=pd.Timestamp(season_starts.get(4, "1978-11-10")),
    end_date=pd.Timestamp(season_starts.get(9, "1983-11-11"))
)

evaluation = env.run()

print(f"Final bankroll: {env.bankroll:.2f}")

history = env.get_history()
print(history)
