#cmd: shiny run --reload --launch-browser  dashboard/app.py

import seaborn as sns
from faicons import icon_svg

# Import data from shared.py
from shared import app_dir, df, nba_games_inseasonn_w_pred, daily_accuracy, season_accuracy

from shiny import reactive
from shiny.express import input, render, ui

ui.page_opts(title="Foreshadownba prediction dashboard", fillable=True)

with ui.sidebar(title="Filter controls"):
    ui.input_slider("mass", "Mass", 2000, 6000, 6000)
    ui.input_checkbox_group(
        "species",
        "Species",
        ["Adelie", "Gentoo", "Chinstrap"],
        selected=["Adelie", "Gentoo", "Chinstrap"],
    )

#TODO: Rename header and all title
with ui.layout_column_wrap(fill=False):
    with ui.value_box(showcase=icon_svg("ruler-horizontal")):
        "Games Correctly Predicted"

        @render.text
        def correctly_predicted():
            return season_accuracy['sum_row_accuracy'].values[0]

    with ui.value_box(showcase=icon_svg("earlybirds")):
        "Total Number of Games"

        @render.text
        def nb_games_total():
            return season_accuracy['row_count'].values[0]

    with ui.value_box(showcase=icon_svg("ruler-vertical")):
        "Inseason Accuracy"

        @render.text
        def inseason_accuracy():
            return f"{round(season_accuracy['season_accuracy']*100,2)[0]}%" #f"{60:.1f}%" #f"{filtered_df()['bill_depth_mm'].mean():.1f} mm"

#TODO: Rename header and all title
with ui.layout_columns():
    with ui.card(full_screen=True):
        ui.card_header("Bill length and depth")

        #TODO: Change plot to display good vs bad results
        @render.plot
        def length_depth():
            return sns.scatterplot(
                data=filtered_df(),
                x="bill_length_mm",
                y="bill_depth_mm",
                hue="species",
            )

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
