import pandas as pd
import numpy as np


def parse_content(value):
    """
    This script works for processing and preparing the items.csv
    This is just fixing the content column to be all integer for now.
    Examples:
    - 10   -> 10
    - 5X10 -> 50
    :param value:
    :return:
    """
    # handle missing values
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    # case like 5X10
    if "X" in value:
        parts = value.split("X")
        result = 1
        for part in parts:
            result *= int(float(part.strip()))
        return result

    # TODO: solve for PAK and L 125 is omitting the values good enough?
    elif 'PAK' in value:
        pass
        return pd.NA
        # do seomthing
    elif 'L   125' in value:
        pass
        return pd.NA
    # case like normal number
    return float(value)

def replace_zero_competitor_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace competitorPrice == 0 with NaN.

    Assumption:
    - A value of 0 indicates no valid competitor price (not a real price)

    Parameters:
        df (pd.DataFrame): Input dataframe with 'competitorPrice'

    Returns:
        pd.DataFrame: Copy of dataframe with 0 replaced by NaN
    """
    df = df.copy()

    if "competitorPrice" not in df.columns:
        raise KeyError("Column 'competitorPrice' not found")

    df["competitorPrice"] = df["competitorPrice"].where(
        df["competitorPrice"] > 0,
        np.nan
    )

    return df

def add_has_competitor_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary feature indicating whether a competitor exists.
    Feature:
    - has_competitor = 1 if competitorPrice is not NaN
    - has_competitor = 0 if competitorPrice is NaN
    """
    df = df.copy()

    if "competitorPrice" not in df.columns:
        raise KeyError("Column 'competitorPrice' not found")

    df["has_competitor"] = df["competitorPrice"].notna().astype(int)

    return df

def normalize_pharmform(value):
    """
    Normalize a single pharmForm value by removing extra surrounding spaces
    and converting all letters to uppercase.

    Missing values are returned unchanged.

    Parameters:
        value:
            A single value from the 'pharmForm' column.

    Returns:
        The cleaned pharmForm value in uppercase, or the original missing value.
    """
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).strip().upper()

def encode_campaign_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the 'campaignIndex' column with three one-hot encoded columns:
    'campaignIndex_A', 'campaignIndex_B', and 'campaignIndex_C'.

    Encoding rules:
    - A  -> 1, 0, 0
    - B  -> 0, 1, 0
    - C  -> 0, 0, 1
    - NA -> 0, 0, 0

    The original 'campaignIndex' column is removed.

    Parameters:
        df (pd.DataFrame):
            Input dataframe containing the 'campaignIndex' column.

    Returns:
        pd.DataFrame:
            A copy of the dataframe with the encoded campaign columns added
            and the original 'campaignIndex' column removed.
    """
    df = df.copy()

    campaign_dummies = pd.get_dummies(df["campaignIndex"], prefix="campaignIndex")
    df = pd.concat([df, campaign_dummies], axis=1)

    for col in ["campaignIndex_A", "campaignIndex_B", "campaignIndex_C"]:
        if col not in df.columns:
            df[col] = 0

    df[["campaignIndex_A", "campaignIndex_B", "campaignIndex_C"]] = (
        df[["campaignIndex_A", "campaignIndex_B", "campaignIndex_C"]].astype(int)
    )

    df = df.drop(columns=["campaignIndex"])

    return df

def difference_competitor_price(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create price comparison features between our price and the competitor price.

        Features created:
        - price_diff_competitor: absolute difference (price - competitorPrice)
            This feature shows the absolute difference between the price and the competitor price.
            for example:
                9 (our price) - 10 (competitor price) = -1
                10 (our price) - 9 (competitor price) = 1
        - price_ratio_competitor: relative ratio (price / competitorPrice)
            - This feature shows ratio of our price / competitor price
            - A value of 1.10 means you are 10% more expensive; 0.90 means you are 10% cheaper.
            - Advantage: allows the model to treat a $10 vs $11 comparison the same as a $100 vs $110 comparison
            - Best for  Model: Random Forests or XGBoost
        - price_pct_diff_competitor: percentage difference relative to competitorPrice
            - The percentage difference relative to the competitor’s price
    """
    df = df.copy()

    # Mask for valid competitor
    valid_mask = df["competitorPrice"].notna()

    # Absolute difference (safe: NaN propagates automatically)
    df["price_diff_competitor"] = (
            df["price"] - df["competitorPrice"]
    ).round(2)

    # Ratio (where competitor exists)
    df["price_ratio_competitor"] = np.where(
        valid_mask,
        (df["price"] / df["competitorPrice"]).round(4),
        np.nan
    )

    # Percentage difference (where competitor exists)
    df["price_pct_diff_competitor"] = np.where(
        valid_mask,
        ((df["price"] - df["competitorPrice"]) / df["competitorPrice"] * 100).round(2),
        np.nan
    )

    return df

def find_frequency_threshold(series: pd.Series, coverage_target: float) -> int:
    """
    Finds the minimum frequency count needed so that the most common
    values in a feature cover the selected share of all rows.
    This can be used to identify rare values for grouping into "other".

    Example:
    If coverage_target = 0.95, the function returns the minimum count
    needed to keep values that together represent about 95% of all
    interactions in the feature.

    :param series: A pandas Series containing the categorical feature.
    :param coverage_target: The desired cumulative coverage level
        between 0 and 1, for example 0.95 for 95%.
    :return: The frequency threshold count. Values with frequency below
        this threshold can be treated as rare.
    """
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    total = counts.sum()
    cumulative_coverage = counts.cumsum() / total
    values_needed = cumulative_coverage[cumulative_coverage <= coverage_target]
    if len(values_needed) == 0:
        return counts.iloc[0]
    threshold = counts.iloc[len(values_needed)]
    return threshold

def group_rare_categories_by_coverage(df: pd.DataFrame, coverage_target: float = 0.95) -> pd.DataFrame:
    """
    Groups rare values in selected high-cardinality categorical features
    into the category "other" based on cumulative frequency coverage.
    This reduces noise from very rare categories while keeping the most
    common values that represent the majority of interactions.

    Example:
    If coverage_target = 0.95, the function keeps the most frequent
    values that together cover about 95% of rows in each selected
    feature, and replaces the remaining rare values with "other".

    :param df: A pandas DataFrame containing the features to process.
    :param coverage_target: The desired cumulative coverage level
        between 0 and 1, for example 0.95 for 95%.
    :return: The updated DataFrame with rare categories grouped into
        "other".
    """
    high_cardinality_features = ["manufacturer", "group", "category"]

    for feature in high_cardinality_features:
        threshold = find_frequency_threshold(df[feature], coverage_target)
        rare_values = set(df[feature].value_counts()[lambda x: x < threshold].index)
        df[feature] = df[feature].apply(lambda x: "other" if x in rare_values else x)

    return df

if __name__ == "__main__":
    pass
