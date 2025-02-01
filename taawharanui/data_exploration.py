import pandas as pd
from trap_analysis.analysis import *
from rich import print
import plotly.express as px

if __name__ == "__main__":
    # lets explore the dataset.

    # load in the traps csv
    traps = pd.read_csv("taawharanui/data/manage_traps.csv")
    traps = convert_columns_to_snake_case(traps)
    print(traps.columns)

    # Let's get a map of different traps
    # create_pie_chart(
    #     df=traps, column_name="trap_type", title="Trap Counts, Taawharanui"
    # )
    create_park_map(df=traps, colour_by="nid", title="Trap placement", save_bool=False)
