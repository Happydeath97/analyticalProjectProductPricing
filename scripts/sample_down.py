import pandas as pd
from sklearn.model_selection import train_test_split

"""
run this script once you have run create_unified_dataset.py

this will take the unified dataset provided and create a sample with 200k rows for better work

the 200k sample will be stored at ../data/processed/sample.csv
"""


def sample_dataset(df: pd.DataFrame, sample_size: int = 200_000, random_state: int = 42) -> pd.DataFrame:
    """
    Create a representative sample from the unified dataset.

    The function first tries to preserve both:
    - the distribution of the target column 'order'
    - the rough time distribution using weeks derived from 'day'

    If stratification by both 'order' and 'week' is not possible because some
    strata are too small, the function falls back to stratification by 'order' only.

    Parameters:
        df (pd.DataFrame):
            The unified input dataset.
        sample_size (int):
            Number of rows to include in the sample. Default is 200_000.
        random_state (int):
            Random seed used for reproducibility. Default is 42.

    Returns:
        pd.DataFrame:
            A sampled dataframe with approximately the same class distribution
            as the original dataset.
    """
    if sample_size >= len(df):
        return df.copy()

    if "order" not in df.columns:
        raise ValueError("Column 'order' is required for stratified sampling.")

    # Try stratifying by both order and week
    if "day" in df.columns:
        sampled_df = df.copy()
        sampled_df["week"] = (sampled_df["day"] // 7).astype(int)
        sampled_df["strata"] = sampled_df["order"].astype(str) + "_" + sampled_df["week"].astype(str)

        strata_counts = sampled_df["strata"].value_counts()
        valid_strata = strata_counts[strata_counts >= 2].index
        sampled_df_valid = sampled_df[sampled_df["strata"].isin(valid_strata)].copy()

        if len(sampled_df_valid) >= sample_size and sampled_df_valid["strata"].nunique() > 1:
            sampled_df, _ = train_test_split(
                sampled_df_valid,
                train_size=sample_size,
                stratify=sampled_df_valid["strata"],
                random_state=random_state
            )
            return sampled_df.drop(columns=["week", "strata"])

    # Fallback: stratify only by order
    sampled_df, _ = train_test_split(
        df,
        train_size=sample_size,
        stratify=df["order"],
        random_state=random_state
    )

    return sampled_df


if __name__ == "__main__":
    unified_df = pd.read_csv("../data/processed/processed_joined_dataset.csv", sep="|")
    print(unified_df.columns.tolist())
    sample_df = sample_dataset(unified_df)
    sample_df.to_csv("../data/processed/sample.csv", sep="|", index=False)
    print(f"Sample with {len(sample_df)} rows saved to ../data/processed/sample.csv")