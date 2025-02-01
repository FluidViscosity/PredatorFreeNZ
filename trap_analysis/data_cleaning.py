def convert_columns_to_snake_case(df):
    """Convert columns to snake case."""
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df


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
