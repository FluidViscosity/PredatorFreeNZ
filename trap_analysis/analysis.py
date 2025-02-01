import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from rich import print
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ephem

import statsmodels.api as sm
from statsmodels.formula.api import ols

from trap_analysis.data_cleaning import convert_columns_to_snake_case
from trap_analysis.plotter import create_line_map, create_park_map, create_pie_chart


"""


function:
for each line, return the trap list including the success rate, the date of the last kill, the most common speciest caught

Geospatial Analysis:

Location Efficiency: Success rate based on geographic coordinates (lat/long, easting/northing).
Heatmaps: Density maps of catches and trap locations.
"""


def anova_preprocess_data(df, variables):
    # Only consider records where the subsequent kill status and all other variables are not null
    required_columns = ["subsequent_kill"] + variables
    df_processed = df.dropna(subset=required_columns)
    return df_processed


def fit_ols(df, model_str):
    """Fit Ordinary Least Squares model."""

    # Fit the model
    model = ols(
        model_str,
        data=df_anova,
    ).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    return anova_table


def get_recorded_period(df):
    """Get the recorded period."""
    try:
        return df["date"].min().strftime("%m-%Y"), df["date"].max().strftime("%m-%Y")
    except AttributeError:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")
        return df["date"].min(), df["date"].max()


def total_traps(df):
    """Total Traps: Count of traps installed."""
    return df["trap_id"].nunique()


def active_traps(df):
    """Active Traps: Number of traps currently active (not retired)."""
    return df[df["retired"] == "Active"]["trap_id"].nunique()


def catch_rate(df):
    """Catch Rate: Number of pests caught per trap over a specified period."""
    return df.groupby("trap_id")["total_kills"].sum().mean()


def trap_type_pie_chart(df):
    """Trap Type: Pie chart of trap types.
    The name of the trap type and the number of traps of that type should be in the segment or shown by a leader line
    """
    trap_type_counts = df["trap_type"].value_counts()
    names = (
        trap_type_counts.index.astype(str) + " (" + trap_type_counts.astype(str) + ")"
    )
    fig = px.pie(
        trap_type_counts,
        values=trap_type_counts.values,
        names=names,
        title="Trap Type Distribution",
    )
    fig.update_layout(legend=dict(x=1, y=0.5, xanchor="right", yanchor="middle"))
    fig.update_traces(
        textposition="inside", textinfo="percent+label", textfont=dict(size=16)
    )

    fig.show()


def kill_efficiency(df):
    """Kill Efficiency: Total kills per trap type."""
    return df.groupby("trap_type")["total_kills"].sum()


def drop_identical_columns(df1, df2, merged_df):
    """Drop identical columns."""
    for col in df1.columns:
        if col in df2.columns:
            try:
                merged_df[col + "_x"].equals(merged_df[col + "_y"])
                merged_df.drop(columns=[col + "_y"], inplace=True)
                merged_df.rename(columns={col + "_x": col}, inplace=True)
            except KeyError:
                pass
    return merged_df


def kill_percentage_for_best_n_pct_traps(df: pd.DataFrame, n: float) -> float:
    """
    Kill Percentage for Best n% Traps: Percentage of total kills from the top n% of traps.
    n: float between 0 and 1.
    """
    top_traps = df.groupby("trap_id")["total_kills"].sum().sort_values(ascending=False)
    top_traps_kills = top_traps.head(int(n * len(top_traps))).sum()
    return (top_traps_kills / df["total_kills"].sum()) * 100


def print_basic_metrics(df):

    print(f"Total traps: {total_traps(df)}")
    print(f"Active traps: {active_traps(df)}")
    print(f"Average catch rate: {catch_rate(df)}")
    print(f"Kills per trap type: {kill_efficiency(df)}")

    print(
        f"The best 10% of traps catch {kill_percentage_for_best_n_pct_traps(df, 0.1):.1f}% of the target kills."
    )
    print(
        f"The best 20% of traps catch {kill_percentage_for_best_n_pct_traps(df, 0.2):.1f}% of the target kills."
    )
    print(
        f"The best 30% of traps catch {kill_percentage_for_best_n_pct_traps(df, 0.3):.1f}% of the target kills."
    )
    print(
        f"The best 40% of traps catch {kill_percentage_for_best_n_pct_traps(df, 0.4):.1f}% of the target kills."
    )

    print(
        f"The best 50% of traps catch {kill_percentage_for_best_n_pct_traps(df, 0.5):.1f}% of the target kills."
    )


def species_information(df):
    """
    Species Information:

    Species Distribution: Breakdown of different species caught.
    Species-Specific Success Rate: Success rate per species.
    """

    # Species Distribution
    species_distribution = df.groupby("species_caught")["total_kills"].sum()
    print(f"Species Distribution: {species_distribution}")
    non_target_species = ["Bird", "Myna"]
    non_target_species_kills = species_distribution[non_target_species].sum()
    target_speciest_kills = species_distribution.drop(non_target_species).sum()
    print(f"Non-Target Species: {non_target_species_kills}")
    print(f"Target Species: {target_speciest_kills}")
    print(
        f"Non-target catch rate: {non_target_species_kills / target_speciest_kills * 100:.1f}%",
    )

    # Species-Specific Success Rate
    species_success_rate = (
        df.groupby("species_caught")["total_kills"].sum()
        / df.groupby("species_caught")["trap_id"].nunique()
    )
    print("Species-Specific Success Rate: ", species_success_rate)
    print("The higher the number, the fewer traps were used to catch the species.")

    return species_distribution, species_success_rate


def temporal_analysis(df) -> pd.DataFrame:
    """
    Temporal Analysis:
    Date In stalled vs. Last Record: Time analysis of trap activity and effectiveness.
    Seasonal Trends: Variation in catch rates by month/season.
    Trap lines: Analysis of trap lines and their effectiveness.
    which lines are most effective. which lines are checked most frequently. mean checking fq, std deviation, etc.
    Trap age: An    alysis of trap age and its impact on success rates.

    trap kills per record
    """
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y %H:%M")

    # Date Installed vs. Last Record
    df["date_installed"] = pd.to_datetime(df["date_installed"])
    df["last_record"] = pd.to_datetime(df["last_record"], format="%d/%m/%Y")
    df["trap_age"] = (df["last_record"] - df["date_installed"]).dt.days / 365.25
    # print("Trap Age: ", df["trap_age"].describe())

    # Trap Lines
    trap_lines = df.groupby("line")["total_kills"].sum()
    # print("Trap Lines: ", trap_lines)

    # Trap Age
    trap_age = df.groupby("trap_age")["total_kills"].sum()
    # print("Trap Age: ", trap_age)

    # Find the last date at which a species was caught for each trap_id
    df["date_of_last_kill"] = df.groupby("trap_id")["date"].transform(
        lambda x: x[df["species_caught"].notna()].max()
    )

    # Calculate days since last kill
    current_date = pd.Timestamp.now()
    df["days_since_last_kill"] = (current_date - df["date_of_last_kill"]).dt.days

    # trap kills per record
    df["records"] = df.groupby("trap_id")["trap_id"].transform("count")
    # df["kills_in_period"] = df.groupby("trap_id")["species_caught"].transform("count")
    df["kills_in_period"] = (
        df[df["species_caught"].isin(["Bird", "Myna"]) == False]
        .groupby("trap_id")["species_caught"]
        .transform("count")
    )

    df_traps = df[
        ["trap_id", "kills_in_period", "days_since_last_kill", "records", "last_record"]
    ].drop_duplicates()

    df_traps["trap_kills_per_record"] = (
        df_traps["kills_in_period"] / df_traps["records"]
    )
    # print("Trap Kills per Record: ", df_traps["trap_kills_per_record"].describe())
    # print("Days since last kill: ", df_traps["days_since_last_kill"].describe())

    # Seasonal Trends
    # get the number of species caught per month. I want a df with 12 rows, one for each month
    # Each row will contain the sum of all species caught in that month, across all traps in that period of time
    df["month"] = df["date"].dt.month
    monthly_catch_rate = df.groupby("month")["species_caught"].count()
    # print("Monthly Catch Rate: ", monthly_catch_rate)

    df_traps["trap_type"] = df["trap_type"].astype("category")

    return df, df_traps


def bait_effectiveness(df):
    """
    Bait Effectiveness: Comparison of different bait types and their effectiveness.

    Trap Condition Impact: Analysis of how trap condition affects success rates.
    Bait Effectiveness: Comparison of different bait types and their effectiveness.
    Trap Type and Subtype Performance: Comparison of different trap types and subtypes.
    Trap Placement: Analysis of trap placement and its impact on success rates.
    Per line success rate: Success rate per line.
    Per trap success rate: Success rate per trap.
    Per bait success rate: Success rate per bait.
    Per trap type success rate: Success rate per trap type.
    Per trap subtype success rate: Success rate per trap subtype.
    Per trap condition success rate: Success rate per trap condition.
    """
    normalised_bait_effectiveness_per_trap_type = (
        df.groupby(["trap_type", "initial_bait"])["species_caught"].count()
        / df.groupby(["trap_type", "initial_bait"])["initial_bait"].count()
    )
    print(
        f"Normalised Bait Effectiveness per Trap Type: {normalised_bait_effectiveness_per_trap_type}"
    )
    print(
        "The best 10 baits per trap type: ",
        normalised_bait_effectiveness_per_trap_type.sort_values(ascending=False).head(
            10
        ),
    )
    norm_bait_effectiveness = (
        df.groupby("initial_bait")["species_caught"].count()
        / df.groupby("initial_bait")["initial_bait"].count()
    )
    print(f"Normalised Bait Effectiveness: {norm_bait_effectiveness}")
    print(
        "The best 10 baits: ",
        norm_bait_effectiveness.sort_values(ascending=False).head(10),
    )


def save_line_to_csv(df_line, line):
    df_line[df_line["line"] == line].to_csv(f"{line}.csv")


def print_line_summary(df_line, line):
    print(f"Line: {line}")
    print(
        df_line[df_line["line"] == line].sort_values(
            "trap_kills_per_record", ascending=False
        )
    )
    print("\n")


def create_line_summary(line, df, big_df):
    df_line = df.copy()
    df_line["trap_kills_per_record"] = df_line["trap_kills_per_record"].round(2)
    df_line["recency"] = df_line["recency"].round(2)
    fig_map1 = create_line_map(
        line,
        df_line[df_line["line"] == line],
        "trap_kills_per_record",
        False,
        date=f"{get_recorded_period(big_df)}",
    )
    # fig_map2 = create_line_map(
    #     line,
    #     df_line[df_line["line"] == line],
    #     "recency",
    #     False,
    #     date=f"{get_recorded_period(big_df)}",
    # )
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Trap name",
                        "Kills in period",
                        "Number of records",
                        "Days since last kill",
                        "Trap kills per record",
                    ],
                ),
                cells=dict(
                    values=[
                        df_line[df_line["line"] == line]["code"],
                        df_line[df_line["line"] == line]["kills_in_period"],
                        df_line[df_line["line"] == line]["records"],
                        df_line[df_line["line"] == line]["days_since_last_kill"],
                        df_line[df_line["line"] == line]["trap_kills_per_record"],
                    ],
                ),
            )
        ]
    )

    with open(f"{line}.html", "w") as f:
        # f.write("<table><tr><td>")
        f.write(fig_map1.to_html(full_html=False, include_plotlyjs="cdn"))
        # f.write("</td><td>")
        # f.write(fig_table.to_html(full_html=False, include_plotlyjs="cdn"))
        # # f.write(fig_map2.to_html(full_html=False, include_plotlyjs="cdn"))
        # f.write("</td></tr><tr><td>")
        # # f.write("</td></tr></table>")


def create_line_summary(line, df, big_df):
    df_line = df.copy()
    df_line["trap_kills_per_record"] = df_line["trap_kills_per_record"].round(2)
    df_line["recency"] = df_line["recency"].round(2)
    fig_map1 = create_line_map(
        line,
        df_line[df_line["line"] == line],
        "trap_kills_per_record",
        False,
        date=f"{get_recorded_period(big_df)}",
    )
    # fig_map2 = create_line_map(
    #     line,
    #     df_line[df_line["line"] == line],
    #     "recency",
    #     False,
    #     date=f"{get_recorded_period(big_df)}",
    # )
    fig_table = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Trap name",
                        "Kills in period",
                        "Number of records",
                        "Days since last kill",
                        "Trap kills per record",
                    ],
                ),
                cells=dict(
                    values=[
                        df_line[df_line["line"] == line]["code"],
                        df_line[df_line["line"] == line]["kills_in_period"],
                        df_line[df_line["line"] == line]["records"],
                        df_line[df_line["line"] == line]["days_since_last_kill"],
                        df_line[df_line["line"] == line]["trap_kills_per_record"],
                    ],
                ),
            )
        ]
    )

    with open(f"{line}.html", "w") as f:
        f.write("<table><tr><td>")
        f.write(fig_map1.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</td><td>")
        f.write(fig_table.to_html(full_html=False, include_plotlyjs="cdn"))
        # f.write(fig_map2.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("</td></tr><tr><td>")
        # f.write("</td></tr></table>")


def find_worst_n_traps(df, n):
    df["is_worst"] = False

    # Get the lines that have more than 5 traps
    lines_with_more_than_n_traps = df_line["line"].value_counts()
    lines_with_more_than_n_traps = lines_with_more_than_n_traps[
        lines_with_more_than_n_traps > n
    ].index

    for line in lines_with_more_than_n_traps:
        worst_n_traps = (
            df_line[df_line["line"] == line]["trap_kills_per_record"].nsmallest(n).index
        )
        df_line.loc[worst_n_traps, "is_worst"] = True


def create_all_maps(df: pd.DataFrame, df_line: pd.DataFrame) -> None:

    create_park_map(df_line, "line", title="Trap lines", save_bool=True)
    create_park_map(
        df_line,
        "records",
        title="Number of records",
        save_bool=True,
        legend_title="Number of <br>records",
    )
    create_park_map(
        df_line,
        "kills_in_period",
        title="Kills in period",
        save_bool=True,
        legend_title="Kills in <br>period",
    )
    create_park_map(
        df_line,
        colour_by="trap_kills_per_record",
        title="Trap kills per record",
        save_bool=True,
        legend_title="Trap kills <br>per record",
    )
    create_park_map(
        df_line,
        "days_since_last_kill",
        title="Days since last kill",
        save_bool=True,
        legend_title="Days since <br>last target kill",
    )
    create_park_map(
        df_line, "recency", title="Most recent target catches", save_bool=True
    )
    for line in df_line["line"].unique():
        create_line_summary(line, df_line, df)
        save_line_to_csv(df_line, line)


if __name__ == "__main__":

    # Load the data
    df1 = pd.read_csv("taawharanui/data/manage_traps.csv")
    df2 = pd.read_csv("taawharanui/data/trap_records.csv")

    df1 = convert_columns_to_snake_case(df1)
    df2 = convert_columns_to_snake_case(df2)

    # Merge datasets on relevant columns
    merged_df = pd.merge(df1, df2, left_on="trap_id", right_on="nid")

    df = drop_identical_columns(df1, df2, merged_df)

    # Print metrics
    print_basic_metrics(df)

    # Species Information
    species_information(df)
    non_target_species = ["Bird", "Myna", "Mouse", "Other", "Unspecified"]
    df_target = df[~df["species_caught"].isin(non_target_species)]
    print("#################")
    print_basic_metrics(df_target)

    # Temporal Analysis
    df_target, df_traps = temporal_analysis(df_target)
    create_pie_chart(df_traps, "trap_type", "Trap Type Distribution")
    create_pie_chart(df_target, "species_caught", "Killed Species Distribution")

    df_outside = df_target[~df_target["line"].str.contains("TL")]

    # Bait Effectiveness
    # bait_effectiveness(df)

    """
    Rank trap lines by total kills
    Rank trap lines by number of records for data period
    Rank trap lines by kills/ num of records (line efficiency)

    for each line, return the trap list including the success rate, 
    the date of the last kill, the most common speciest caught
    # """

    df_line = df_target[
        [
            "trap_id",
            "code",
            "lat",
            "long",
            "line",
            "kills_in_period",
            "days_since_last_kill",
            "records",
        ]
    ].drop_duplicates()
    df_line["trap_kills_per_record"] = df_line["kills_in_period"] / df_line["records"]
    df_line["trap_kills_per_record"] = df_line["trap_kills_per_record"].fillna(0)
    df_line["kills_in_period"] = df_line["kills_in_period"].fillna(0)
    df_line["days_since_last_kill"] = df_line["days_since_last_kill"].fillna(np.inf)
    df_line["records"] = df_line["records"].fillna(0)
    df_line["recency"] = 1 / df_line["days_since_last_kill"]

    # find_worst_n_traps(df, 5)

    create_all_maps(df, df_line)

    shakespear_lat = df_line["lat"].mean()
    shakespear_long = df_line["long"].mean()
    observer = ephem.Observer()
    observer.lat = str(shakespear_lat)
    observer.long = str(shakespear_long)
    observer.elev = 0

    df_moon = df.copy()
    df_moon["moon_phase_when_set"] = df.apply(
        lambda x: ephem.Moon(x["date"]).phase / 100, axis=1
    )
    # print(df_moon["moon_phase_when_set"].describe())
    # Sort by trap_id and date
    df_moon = df_moon.sort_values(by=["trap_id", "date"])

    # Create a lagged column for species caught
    df_moon["subsequent_kill"] = (
        df_moon.groupby("trap_id")["species_caught"].shift(-1).notna().astype(int)
    )

    # print(df_moon)
    # print("Moon phase when set: ", df_moon["moon_phase_when_set"].describe())
    # print("Subsequent kill: ", df_moon["subsequent_kill"].describe())

    df_moon["trap_type"] = df_moon["trap_type"].astype("category")
    df_moon["recorded_by"] = df_moon["recorded_by"].astype("category")
    df_moon["species_caught"] = df_moon["species_caught"].astype("category")
    df_moon["initial_bait"] = df_moon["initial_bait"].astype("category")
    df_moon["month"] = df_moon["month"].astype("category")
    df_moon["line"] = df_moon["line"].astype("category")
    df_moon["trap_condition"] = df_moon["trap_condition"].astype("category")

    # initial_variables = [
    #     "moon_phase_when_set",
    #     "recorded_by",
    #     "month",
    #     "trap_type",
    #     "initial_bait",
    #     "species_caught",
    #     "line",
    #     "trap_condition",
    # ]
    # least significant variables: species caught
    variables = [
        "moon_phase_when_set",
        "recorded_by",
        "month",
        "trap_type",
        "initial_bait",
        "species_caught",
        "line",
        "trap_condition",
    ]
    # df_anova = anova_preprocess_data(df_moon, variables)
    # model_str = "subsequent_kill ~ " + " + ".join(variables)
    # moon_anova = fit_ols(df_anova, model_str)
    # print(moon_anova)

    # # Print the mean subsequent kill rate for each variable
    print(
        df_moon.groupby("recorded_by")["subsequent_kill"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    # print(df_moon.groupby(["line", "recorded_by"])["subsequent_kill"].mean())
    # # print(df_moon.groupby("trap_type")["subsequent_kill"].mean())
    # print(df_moon.groupby("initial_bait")["subsequent_kill"].mean())
    # print(df_moon.groupby("species_caught")["subsequent_kill"].mean())
    # print(df_moon.groupby("line")["subsequent_kill"].mean())
    # # print(df_moon.groupby("trap_condition")["subsequent_kill"].mean())
    print(df_moon.groupby("month")["subsequent_kill"].mean())
    # print(df_moon.groupby("moon_phase_when_set")["subsequent_kill"].mean())

    # Print the mean subsequent kill rate for each variable but exclude mouse, bird and myna
    df_moon_target = df_moon[
        df_moon["species_caught"].isin(non_target_species) == False
    ]
    df_moon_target["subsequent_kill_target"] = df_moon_target.groupby(
        ["trap_id", "date"]
    )["subsequent_kill"].shift(-1)
    # df_moon_target.to_clipboard()
    print(
        df_moon_target.groupby("recorded_by")["subsequent_kill_target"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    print(
        df_moon_target.groupby(["line", "recorded_by"])["subsequent_kill_target"]
        .mean()
        .dropna()
    )
    # print(df_moon_target.groupby("trap_type")["subsequent_kill_target"].mean())
    print(
        df_moon_target.groupby("initial_bait")["subsequent_kill_target"]
        .mean()
        .dropna()
        .sort_values(ascending=False)
    )
    # print(df_moon_target.groupby("species_caught")["subsequent_kill_target"].mean())
    print(df_moon_target.groupby("line")["subsequent_kill_target"].mean().dropna())
    # print(
    #     df_moon_target.groupby("trap_condition")["subsequent_kill_target"]
    #     .mean()
    #     .dropna()
    # )
    print(df_moon_target.groupby("month")["subsequent_kill_target"].mean())
    # print(
    #     df_moon_target.groupby("moon_phase_when_set")["subsequent_kill_target"]
    #     .mean()
    #     .dropna()
    # )
    df_moon_target.groupby("species_caught")[
        "initial_bait"
    ].value_counts().dropna().sort_index(ascending=False).to_clipboard()

    print(
        df_moon_target.groupby("species_caught")["initial_bait"]
        .value_counts()
        .dropna()
        .sort_index(ascending=False)
        .head(15)
    )
    print(
        df_moon.groupby("species_caught")["initial_bait"]
        .value_counts()
        .dropna()
        .sort_values(ascending=False)
        .head(15)
    )

    bins = [-0.01, 0.25, 0.5, 0.75, 1.01]
    labels = ["New Moon", "First Quarter", "Full Moon", "Last Quarter"]

    # Bin the moon_phase_when_set variable
    df_moon_target["moon_phase_category"] = pd.cut(
        df_moon_target["moon_phase_when_set"], bins=bins, labels=labels
    )

    # Find the most common moon phase for subsequent kills
    most_common_moon_phase = df_moon_target[
        df_moon_target["subsequent_kill_target"] == 1
    ]["moon_phase_category"].mode()[0]
    print(most_common_moon_phase)
    print("done")
