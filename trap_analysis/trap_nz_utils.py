def convert_columns_to_snake_case(df):
    """Convert columns to snake case."""
    df.columns = df.columns.str.replace(" ", "_").str.lower()
    return df
