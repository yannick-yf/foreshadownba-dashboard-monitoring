from pathlib import Path
import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def prepare_nba_games_data(file_path):
    """
    Load and prepare NBA games data for analysis.

    Args:
        file_path (str or Path): Path to the NBA games CSV file.

    Returns:
        pd.DataFrame: Prepared DataFrame with additional columns and formatted dates.
    """
    nba_games = pd.read_csv(file_path)
    nba_games = nba_games[['id', 'id_season', 'game_date', 'tm', 'opp', 'results', 'prediction_value']]
    nba_games['row_accuracy'] = np.where(
        nba_games['results'] == nba_games['prediction_value'], 
        1, 
        0
    )
    nba_games['game_date'] = pd.to_datetime(nba_games['game_date'])
    return nba_games

def calculate_daily_accuracy(nba_games):
    """
    Calculate daily accuracy from NBA games data.

    Args:
        nba_games (pd.DataFrame): Prepared NBA games DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing daily accuracy.
    """
    daily_accuracy = nba_games.groupby('game_date').agg(
        row_count=('id', 'count'),
        sum_row_accuracy=('row_accuracy', 'sum')
    ).reset_index()
    daily_accuracy['daily_accuracy'] = daily_accuracy['sum_row_accuracy'] / daily_accuracy['row_count']
    return daily_accuracy

def calculate_season_accuracy(nba_games):
    """
    Calculate season accuracy from NBA games data.

    Args:
        nba_games (pd.DataFrame): Prepared NBA games DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing season accuracy.
    """
    season_accuracy = nba_games.groupby('id_season').agg(
        row_count=('id', 'count'),
        sum_row_accuracy=('row_accuracy', 'sum')
    ).reset_index()
    season_accuracy['season_accuracy'] = season_accuracy['sum_row_accuracy'] / season_accuracy['row_count']
    return season_accuracy

# Constants
app_dir = Path(__file__).parent
nba_games_inseasonn_w_pred = prepare_nba_games_data("./dashboard/nba_games_inseasonn_w_pred.csv")
daily_accuracy = calculate_daily_accuracy(nba_games_inseasonn_w_pred)
season_accuracy = calculate_season_accuracy(nba_games_inseasonn_w_pred)
