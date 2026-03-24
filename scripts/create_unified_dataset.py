import pandas as pd

from utils import (
    parse_content,
    replace_zero_competitor_price,
    add_has_competitor_feature,
    normalize_pharmform,
    encode_campaign_index,
    difference_competitor_price,
    add_is_post_shift
    group_rare_categories_by_coverage
)

if '__main__' == __name__:
    items = pd.read_csv("../data/raw/items.csv", sep="|")
    train = pd.read_csv("../data/raw/train.csv", sep="|")

    item_features = ["pid", "manufacturer", "group", "content", "unit", "pharmForm",
                     "genericProduct", "salesIndex", "category", "campaignIndex", "rrp"]
    new_df = train.merge(items[item_features], on="pid", how="left")

    # Here is the space for computing new features, or for example creating onehot encoding
    # first all one-liners that work/change the whole df, then oneliners that work with 1 feature only
    # if change is too complex make a function that inputs a df and return new df like so:
    new_df = replace_zero_competitor_price(new_df)
    new_df = add_has_competitor_feature(new_df)
    new_df = encode_campaign_index(new_df)
    new_df = difference_competitor_price(new_df)
    new_df = add_is_post_shift(new_df)
    new_df = group_rare_categories_by_coverage(new_df)

    # single feature cleaning ----- This is where we can apply single feature functions
    new_df["content"] = new_df["content"].apply(parse_content)
    new_df = new_df[new_df["content"].notna()]

    new_df["pharmForm"] = new_df["pharmForm"].apply(normalize_pharmform)

    print(new_df.isna().sum().sort_values(ascending=False))

    # Save complete processed dataset
    new_df.to_csv("../data/processed/processed_joined_dataset.csv", sep="|", index=False)
