from pathlib import Path
import numpy as np
import pandas as pd

app_dir = Path(__file__).parent
df = pd.read_csv(app_dir / "penguins.csv")

nba_games_inseasonn_w_pred = pd.read_csv("./dashboard/nba_games_inseasonn_w_pred.csv")
nba_games_inseasonn_w_pred = nba_games_inseasonn_w_pred[['id',	'id_season',	'game_date',	'tm',	'opp',	'results',	'prediction_value']]
nba_games_inseasonn_w_pred['row_accuracy'] = np.where(
    nba_games_inseasonn_w_pred['results'] == nba_games_inseasonn_w_pred['prediction_value'], 
    1, 
    0)
nba_games_inseasonn_w_pred['game_date'] = pd.to_datetime(nba_games_inseasonn_w_pred['game_date'])


# Daily Accuracy
daily_accuracy = nba_games_inseasonn_w_pred.groupby('game_date').agg(
    row_count=('id', 'count'),
    sum_row_accuracy=('row_accuracy', 'sum')
).reset_index()

daily_accuracy['daily_accuracy'] = daily_accuracy['sum_row_accuracy'] / daily_accuracy['row_count']

# Season Accuracy
season_accuracy = nba_games_inseasonn_w_pred.groupby('id_season').agg(
    row_count=('id', 'count'),
    sum_row_accuracy=('row_accuracy', 'sum')
).reset_index()

season_accuracy['season_accuracy'] = season_accuracy['sum_row_accuracy'] / season_accuracy['row_count']


# Load data via boto3
