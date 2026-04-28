

def aggregate_results(df):
    """
    Aggregates the results in the given DataFrame by computing the mean and standard error 
    for each combination of 'num_sequence' and 'metric'.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the columns 'num_sequence', 'metric', and 'value'.

    Returns:
    pandas.DataFrame: A DataFrame with the aggregated results, containing the columns 'num_sequence', 
                      'metric', 'mean', and 'std_error'.
    """
    aggregated_df = df.groupby(["num_sequence", "metric"], as_index=False).agg(
        mean=("value", "mean"), std_error=("value", lambda x: x.std() / (len(x) ** 0.5))
    )
    return aggregated_df