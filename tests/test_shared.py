import pytest
import os
import pandas as pd

from pathlib import Path

from dashboard.shared import (
    setup_aws_credentials,
    load_data,
    load_data_from_s3,
    prepare_nba_games_data,
    calculate_daily_accuracy,
    calculate_season_accuracy
)
from io import StringIO

# Sample data for testing
SAMPLE_NBA_GAMES ="""id,id_season,game_date,tm,opp,results,prediction_value,pred_results_1_line_game,prediction_proba_df_win,prediction_proba_df_loose,prediction_proba_df_win_opp,prediction_proba_df_loose_opp
2024-10-25_CHO_ATL,2025,2024-10-25,ATL,CHO,1,1,1,0.6749312,0.32506877,0.3584256,0.6415744
2024-10-27_ATL_OKC,2025,2024-10-27,ATL,OKC,0,0,0,0.3932142,0.6067858,0.574674,0.425326
2024-10-28_WAS_ATL,2025,2024-10-28,ATL,WAS,0,0,1,0.4801254,0.5198746,0.45842505,0.54157495
2024-10-30_ATL_WAS,2025,2024-10-30,ATL,WAS,0,0,0,0.43977463,0.56022537,0.5786276,0.4213724
2024-11-01_SAC_ATL,2025,2024-11-01,ATL,SAC,0,1,1,0.5302031,0.4697969,0.49553776,0.50446224
"""

setup_aws_credentials()

def test_load_data_from_s3():
    bucket_name = 'foreshadownba'
    file_key = 'inference-pipeline-output/nba_games_inseasonn_w_pred.csv'
    df = load_data_from_s3(bucket_name, file_key)
    assert isinstance(df, pd.DataFrame), "Loaded data is not a DataFrame"
    assert not df.empty, "DataFrame is empty"
    assert df.shape[1] == 12

def test_load_data():
    csv_data = StringIO(SAMPLE_NBA_GAMES)
    df = load_data(csv_data)
    assert isinstance(df, pd.DataFrame), "Loaded data is not a DataFrame"
    assert not df.empty, "DataFrame is empty"
    assert df.shape[1] == 12

def test_prepare_nba_games_data():
    data = pd.read_csv(StringIO(SAMPLE_NBA_GAMES))
    prepared_df = prepare_nba_games_data(data)

    assert isinstance(prepared_df, pd.DataFrame), "Prepared data is not a DataFrame"
    assert "row_accuracy" in prepared_df.columns, "row_accuracy column is missing"
    assert pd.api.types.is_datetime64_any_dtype(prepared_df["game_date"]), "game_date column is not datetime"
    assert prepared_df.loc[0, "row_accuracy"] == 1, "row_accuracy calculation is incorrect"
    assert prepared_df.shape[1] == 8


def test_calculate_daily_accuracy():
    data = pd.read_csv(StringIO(SAMPLE_NBA_GAMES))
    prepared_df = prepare_nba_games_data(data)
    daily_accuracy_df = calculate_daily_accuracy(prepared_df)

    assert isinstance(daily_accuracy_df, pd.DataFrame), "Daily accuracy data is not a DataFrame"
    assert "daily_accuracy" in daily_accuracy_df.columns, "daily_accuracy column is missing"
    assert daily_accuracy_df.loc[daily_accuracy_df["game_date"] == pd.Timestamp("2024-10-28"), "daily_accuracy"].iloc[0] == 1.0, "Daily accuracy calculation is incorrect"

def test_calculate_season_accuracy():
    data = pd.read_csv(StringIO(SAMPLE_NBA_GAMES))
    prepared_df = prepare_nba_games_data(data)
    season_accuracy_df = calculate_season_accuracy(prepared_df)

    assert isinstance(season_accuracy_df, pd.DataFrame), "Season accuracy data is not a DataFrame"
    assert "season_accuracy" in season_accuracy_df.columns, "season_accuracy column is missing"
    assert season_accuracy_df.loc[:, "season_accuracy"].iloc[0] == 0.8, "Season accuracy calculation is incorrect"
