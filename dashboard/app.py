#cmd: shiny run --reload --launch-browser  dashboard/app.py
# List icon: https://fontawesome.com/v6/icons/circle-check?f=classic&s=solid
import seaborn as sns
from faicons import icon_svg
import pandas as pd
from shared import app_dir, nba_games_inseasonn_w_pred, daily_accuracy, season_accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from shiny import reactive
from shiny.express import input, render, ui

ui.page_opts(title="Foreshadownba prediction dashboard", fillable=True)

with ui.sidebar(title="Filter controls"):
    ui.input_select(
        "team",
        "Filter by a team:",
        {
            "All": "All teams",
            "East Conference": {
                "ATL": "Atlanta Hawks", 
                "BOS": "Boston Celtics", 
                "BRK": "Brooklyn Nets",
                "CHI": "Chicago Bulls", 
                "CHO": "Charlotte Hornets", 
                "IND": "Indiana Pacers",
                "DET": "Detroit Pistons",
                "CLE": "Cleveland Cavaliers",
                "MIA": "Miami Heat",
                "NYK": "New York Knicks",
                "ORL": "Orlando Magic",
                "MIL": "Milwakee Bucks",
                "PHI": "Phidadelphie Sixers",
                "WAS": "Washington Wizards",
                "TOR": "Toronto Raptors"
                },
            "West Conference": {
                "DAL": "Dallas Mavericks", 
                "GSW": "Golden State Warriors", 
                "DEN": "Denver Nuggets",
                "HOU": "Houston Rockets",
                "LAL": "Los Angeles Lakers",
                "LAC": "Los Angeles Clippers",
                "MEM": "Memphis Grizzlies",
                "NOP": "New Orleans Pelicans",
                "MIN": "Minnesota Timberwolves",
                "OKC": "Oklahoma City Thunders",
                "SAC": "Sacramento Kings",
                "PHO": "Phoenix Suns",
                "POR": "Portland Trail Blazers",
                "SAS": "San Antonio Spurs",
                "UTA": "Utah Jazz"
                },
        },
    selected='All',
    selectize=True,
    multiple=True,
    )

with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg(name="circle-check",style="solid")):
        "Games Correctly Predicted"

        @render.text
        def correctly_predicted():
            return season_accuracy['sum_row_accuracy'].values[0]

    with ui.value_box(showcase=icon_svg(name="circle-check",style="regular")):
        "Total Number of Games"

        @render.text
        def nb_games_total():
            return season_accuracy['row_count'].values[0]

    with ui.value_box(showcase=icon_svg(name="bullseye",style="solid")):
        "Inseason Accuracy"

        @render.text
        def inseason_accuracy():
            return f"{round(season_accuracy['season_accuracy']*100,2)[0]}%"

with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Confusion Matrix of NBA Games Prediction")

        @render.plot
        def confusion_matrix_shiny():
            cm = confusion_matrix(
                filtered_df()['results'], 
                filtered_df()['prediction_value']
            )

            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm
            )
            disp.plot(cmap=plt.cm.Blues)

            return plt.gcf()

    with ui.card(full_screen=True):
        ui.card_header("NBA games data")

        @render.data_frame
        def summary_statistics():
            render_df = filtered_df()[['id_season', 'game_date', 'tm', 'opp', 'results', 'prediction_value']].copy()
            render_df['game_date'] = render_df['game_date'].astype(str).str.slice(0, 10)
            return render.DataGrid(render_df.sort_values(by='game_date', ascending=False), filters=True)

ui.include_css(app_dir / "styles.css")

@reactive.calc
def filtered_df():
    if list(input.team()) == ['All']:
        filt_df = nba_games_inseasonn_w_pred.copy()
    else:
        filt_df = nba_games_inseasonn_w_pred[
            nba_games_inseasonn_w_pred["tm"].isin(list(input.team())) |
            nba_games_inseasonn_w_pred["opp"].isin(list(input.team()))
            ].copy()

    return filt_df
