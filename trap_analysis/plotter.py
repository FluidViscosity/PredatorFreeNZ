import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def create_pie_chart(df: pd.DataFrame, column_name: str, title: str):
    """
    Pie chart of any countable dataframe column.
    """
    counts = df[column_name].value_counts()
    names = counts.index.astype(str) + " (" + counts.astype(str) + ")"
    fig = px.pie(
        counts,
        values=counts.values,
        names=names,
        title=title,
    )
    fig.update_layout(legend=dict(x=1, y=0.5, xanchor="right", yanchor="middle"))
    fig.update_traces(
        textposition="inside", textinfo="percent+label", textfont=dict(size=16)
    )

    fig.show()


def create_line_map(
    line: str,
    df: pd.DataFrame,
    colour_by: str,
    save_bool: bool = False,
    date: str = None,
):
    human_readable = (colour_by + "_" + date).replace("_", " ").title()

    fig_map = px.scatter_mapbox(
        df,
        title=f"{line}. {human_readable}",
        lat="lat",
        lon="long",
        zoom=15,
        color=colour_by,
        size=colour_by,
        color_continuous_scale=px.colors.sequential.Viridis,
        range_color=[0, 1],
        hover_name="code",
        hover_data={
            "trap_kills_per_record": ":.1f",
            "days_since_last_kill": ":1f",
            "kills_in_period": ":.0f",
            "records": ":.0f",
        },
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 2, "b": 2})
    if save_bool:
        fig_map.write_html("trap_efficiency_map.html")
    # fig_map.show()
    return fig_map


def create_park_map(
    df,
    colour_by="species_caught",
    title: str = None,
    save_bool: bool = False,
    legend_title=None,
):
    size = colour_by
    if title is None:
        title = "Shakespear Park"
    if colour_by == "line":
        size = [5] * len(df)

    fig_map = px.scatter_mapbox(
        df,
        title=title,
        lat="lat",
        lon="long",
        zoom=13.7,  # -36.606765, 174.810414
        center={"lat": -36.606250, "lon": 174.807127},
        color=colour_by,
        size=size,
        color_continuous_scale=px.colors.sequential.Viridis,
        # range_color=[0, 1],
        hover_name="code",
        hover_data={
            "trap_kills_per_record": ":.1f",
            "days_since_last_kill": ":1f",
            "kills_in_period": ":.0f",
            "records": ":.0f",
        },
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    fig_map.update_layout(margin={"r": 0, "t": 40, "l": 2, "b": 2})

    if legend_title:
        fig_map.update_layout(coloraxis_colorbar_title_text=legend_title)

    if save_bool:
        fig_map.write_html(f"shakespear_park_map_{colour_by}.html")
    fig_map.show()
