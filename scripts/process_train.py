import pandas as pd

train = pd.read_csv("data/raw/train.csv", sep = "|")

# Create bins
train['price_bin'] = pd.cut(train['price'], bins=50, include_lowest=True)

# Statistics from non-missing competitor prices
bin_stats = (
    train[train['competitorPrice'].notna()]
    .groupby('price_bin', observed=True)
    .agg(
        avg_price=('price', 'mean'),
        avg_competitor=('competitorPrice', 'mean')
    )
)

# Percentage difference
bin_stats['pct_diff'] = (
    (bin_stats['avg_competitor'] - bin_stats['avg_price'])
    / bin_stats['avg_price']
) * 100

# Missing rows
missing_mask = train['competitorPrice'].isna()

missing_prices = train.loc[missing_mask, ['price', 'price_bin']].copy()

# Merge pct_diff
missing_prices = missing_prices.merge(
    bin_stats[['pct_diff']],
    left_on='price_bin',
    right_index=True,
    how='left'
)

# Estimate missing competitorPrice
missing_prices['estimated_competitorPrice'] = (
    missing_prices['price']
    + (missing_prices['price'] / 100 * missing_prices['pct_diff'])
)

# Fill back into original dataframe
train.loc[missing_mask, 'competitorPrice'] = missing_prices['estimated_competitorPrice'].values

# Fallback for any still-missing values
overall_median = train['competitorPrice'].median()
train['competitorPrice'] = train['competitorPrice'].fillna(overall_median)

# Drop helper column
train = train.drop(columns=['price_bin'])

# Save complete processed dataset
train.to_csv("data/processed/processedTrain.csv", sep= "|", index=False)

print("File saved: data/processed/processedTrain.csv")
print("Remaining missing competitorPrice values:", train['competitorPrice'].isna().sum())