import pandas as pd
from trap_analysis.analysis import *
from rich import print
import plotly.express as px

if __name__ == "__main__":
    # lets explore the dataset.

    # load in the traps csv
    traps = pd.read_csv("taawharanui/data/traps_overview.csv")
    traps = convert_columns_to_snake_case(traps)
    # print(traps.columns)

    # Let's get a map of different traps
    # create_pie_chart(
    #     df=traps, column_name="trap_type", title="Trap Counts, Taawharanui"
    # )
    # create_park_map(df=traps, colour_by="nid", title="Trap placement", save_bool=False)
    # assign id for each unique installer
    # traps["installer_id"] = traps["installed_by"].astype("category").cat.codes
    # traps["installer_id"].replace(-1, 99, inplace=True)
    # create_park_map(
    #     df=traps[traps["trap_type"] == "DOC 200"],
    #     colour_by="installer_id",
    #     title="DOC 200 Trap placement",
    #     save_bool=False,
    # )

    # lets look at the trap records
    records = pd.read_csv("taawharanui/data/trap_records2.csv")
    records = convert_columns_to_snake_case(records)
    records.rename(columns={"trap_nid": "trap_id"}, inplace=True)

    # Merge datasets on relevant columns
    merged_df = pd.merge(records, traps, left_on="trap_id", right_on="nid")

    df = drop_identical_columns(traps, records, merged_df)

    # Print metrics
    print_basic_metrics(df)

    # Species Information
    species_information(df)
