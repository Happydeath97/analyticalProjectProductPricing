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
    return int(float(value))


def fill_missing_competitor_price(df: pd.DataFrame) -> pd.DataFrame:

    # TODO: CompetitorPrice has many 0.00.This is an error/flaw in the file and needs to be cleaned!!!!
    """
    This is a function to replace Said's process_train.py
    Fill missing values in the 'competitorPrice' column using price-bin-based estimation.

    The method groups rows into bins according to the 'price' column, then calculates
    the average percentage difference between 'price' and 'competitorPrice' for rows
    where 'competitorPrice' is available. Missing competitor prices are estimated
    using the average percentage difference of the corresponding price bin.

    If a missing value still cannot be estimated, for example because its bin has
    no available competitor price statistics, the method fills it with the overall
    median of the 'competitorPrice' column.

    Parameters:
        df (pd.DataFrame):
            Input dataframe containing at least the columns:
            - 'price'
            - 'competitorPrice'

    Returns:
        pd.DataFrame:
            A copy of the input dataframe where missing values in
            'competitorPrice' have been filled.

    Raises:
        KeyError:
            If required columns such as 'price' or 'competitorPrice' are missing.
    """
    df = df.copy()

    # create bins from price
    df["price_bin"] = pd.cut(df["price"], bins=50, include_lowest=True)

    # compute average price and competitorPrice for each bin
    bin_stats = (
        df[df["competitorPrice"].notna()]
        .groupby("price_bin", observed=True)
        .agg(
            avg_price=("price", "mean"),
            avg_competitor=("competitorPrice", "mean")
        )
    )

    # compute average percentage difference in each bin
    bin_stats["pct_diff"] = (
        (bin_stats["avg_competitor"] - bin_stats["avg_price"])
        / bin_stats["avg_price"]
    ) * 100

    # find rows where competitorPrice is missing
    missing_mask = df["competitorPrice"].isna()

    missing_prices = df.loc[missing_mask, ["price", "price_bin"]].copy()

    # attach pct_diff from the matching price bin
    missing_prices = missing_prices.merge(
        bin_stats[["pct_diff"]],
        left_on="price_bin",
        right_index=True,
        how="left"
    )

    # estimate missing competitorPrice
    missing_prices["estimated_competitorPrice"] = round(
        missing_prices["price"] +
        (missing_prices["price"] / 100 * missing_prices["pct_diff"]),
        2
    )

    # write estimated values back
    df.loc[missing_mask, "competitorPrice"] = missing_prices["estimated_competitorPrice"].values

    # fallback if some values are still missing
    overall_median = df["competitorPrice"].median()
    df["competitorPrice"] = df["competitorPrice"].fillna(overall_median)

    # remove helper column
    df = df.drop(columns=["price_bin"])

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
        return value
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

    required_cols = ["price", "competitorPrice"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    competitor_safe = df["competitorPrice"].replace(0, np.nan)

    # Use .round() as a method on the Series, not as a wrapper function
    df["price_diff_competitor"] = (df["price"] - df["competitorPrice"]).round(2)
    df["price_ratio_competitor"] = (df["price"] / competitor_safe).round(2)

    df["price_pct_diff_competitor"] = (
            ((df["price"] - df["competitorPrice"]) / competitor_safe) * 100
    ).round(2)

    return df

if __name__ == "__main__":
    pass
