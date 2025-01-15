from pathlib import Path
import numpy as np
import pandas as pd
import boto3
from io import StringIO
import os

def load_data_from_s3(bucket_name, file_key):
    """
    Load a CSV file from an S3 bucket into a pandas DataFrame.

    Args:
        bucket_name (str): Name of the S3 bucket.
        file_key (str): Key (path) of the file in the S3 bucket.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    # Initialize the S3 client
    s3_client = boto3.client('s3')

    # Download the file from S3 into a memory buffer
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    file_content = response['Body'].read().decode('utf-8')

    # Read the CSV content into a pandas DataFrame
    return pd.read_csv(StringIO(file_content))

def load_data(file_path):
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(file_path)

def prepare_nba_games_data(nba_games: pd.DataFrame) -> pd.DataFrame:
    """
    Load and prepare NBA games data for analysis.

    Args:
        file_path (str or Path): Path to the NBA games CSV file.

    Returns:
        pd.DataFrame: Prepared DataFrame with additional columns and formatted dates.
    """
    nba_games = nba_games[['id', 'id_season', 'game_date', 'tm', 'opp', 'results', 'prediction_value']]
    nba_games['row_accuracy'] = np.where(
        nba_games['results'] == nba_games['prediction_value'], 
        1, 
        0
    )
    nba_games['game_date'] = pd.to_datetime(nba_games['game_date'])
    return nba_games

def calculate_daily_accuracy(nba_games: pd.DataFrame) -> pd.DataFrame:
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

def calculate_season_accuracy(nba_games: pd.DataFrame) -> pd.DataFrame:
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
bucket_name = 'foreshadownba'
file_key = 'inference-pipeline-output/nba_games_inseasonn_w_pred.csv'
# my_profile_name = 'ipfy'
# os.environ['AWS_PROFILE'] = my_profile_name

# Get Data
nba_games_inseasonn_w_pred = prepare_nba_games_data(load_data_from_s3(bucket_name, file_key))
daily_accuracy = calculate_daily_accuracy(nba_games_inseasonn_w_pred)
season_accuracy = calculate_season_accuracy(nba_games_inseasonn_w_pred)
