#cmd: shiny run --reload --launch-browser  dashboard/app.py
# List icon: https://fontawesome.com/v6/icons/circle-check?f=classic&s=solid
import seaborn as sns
from faicons import icon_svg

# Import data from shared.py
from shared import app_dir, nba_games_inseasonn_w_pred, daily_accuracy, season_accuracy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from shiny import reactive
from shiny.express import input, render, ui

ui.page_opts(title="Foreshadownba prediction dashboard", fillable=True)

# with ui.sidebar(title="Filter controls"):
#     ui.input_slider("mass", "Mass", 2000, 6000, 6000)
#     ui.input_checkbox_group(
#         "species",
#         "Species",
#         ["Adelie", "Gentoo", "Chinstrap"],
#         selected=["Adelie", "Gentoo", "Chinstrap"],
#     )

#TODO: Rename header and all title
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

#TODO: Rename header and all title
with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Confusion Matrix of NBA Games Prediction")

        #TODO: Change plot to display good vs bad results
        @render.plot
        def confusion_matrix_shiny():
            cm = confusion_matrix(
                nba_games_inseasonn_w_pred['results'], 
                nba_games_inseasonn_w_pred['prediction_value']
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
            return render.DataGrid(nba_games_inseasonn_w_pred, filters=True)


ui.include_css(app_dir / "styles.css")

#TODO: Define filter on inseason pred data: Exemple: team, date
@reactive.calc
def filtered_df():
    filt_df = df[df["species"].isin(input.species())]
    filt_df = filt_df.loc[filt_df["body_mass_g"] < input.mass()]
    return filt_df
