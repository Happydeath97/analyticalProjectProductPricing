# Data Cleaning Comparison: My Approach vs Yours krystof

This document compares the data preparation decisions I made in
`02_data_preparation.ipynb` against your approach in
`create_unified_dataset.py` and `utils.py`. For each difference I explain
what I did, what you did, and why I went a different way.

---

## 1. Missing competitorPrice Values

**`scripts/utils.py`** estimates missing competitor prices using price-bin
statistics. You group rows into 50 price bins, compute the average percentage
difference between price and competitorPrice within each bin, then use that
percentage to estimate the missing value. If a bin has no data, the overall
median is used as a fallback.

```python
df["price_bin"] = pd.cut(df["price"], bins=50, include_lowest=True)
bin_stats = (
    df[df["competitorPrice"].notna()]
    .groupby("price_bin", observed=True)
    .agg(avg_price=("price", "mean"), avg_competitor=("competitorPrice", "mean"))
)
bin_stats["pct_diff"] = (
                                (bin_stats["avg_competitor"] - bin_stats["avg_price"]) / bin_stats["avg_price"]
                        ) * 100
```

### My approach

I kept missing competitor prices as NaN and added a binary flag to preserve
the missingness signal. A binary flag here means a new column that is 1 when
competitorPrice was missing and 0 when it was present. The model sees this
as a feature just like any other, so it can learn that "no competitor data
available" is itself a signal.

```python
df["competitor_price_missing"] = df["competitorPrice"].isna().astype(int)
```

### Difference

Your approach assumes that a missing competitor price can be approximated
from products in the same price range. That assumption may not hold:
competitor data might be missing because the product has no direct
competitor at all, not because the data entry failed. Imputing a fabricated
value destroys that signal entirely.

The way I thought about it, the binary flag approach is more honest. It
tells the model two things: what the competitor price was when available,
and whether competitor data was even present at all. Tree-based models like
CatBoost and XGBoost handle NaN in numeric features natively, so no
imputation was needed for the models I chose. I only found this out while
exploring which models to use, so this decision came later rather than
upfront.

---

## 2. competitorPrice Zeros

**`scripts/utils.py`** replaces zeros using the same bin-based estimation
as above.

```python
error_mask = df["competitorPrice"] == 0
df.loc[error_mask, "pct_diff"] = df.loc[error_mask, "price_bin"].map(bin_stats["pct_diff"])
df.loc[error_mask, "competitorPrice"] = round(
    df.loc[error_mask, "price"] * (1 + df.loc[error_mask, "pct_diff"]), 2
)
```

### My approach

I recoded zeros to NaN. A price of zero is not a real price and the 976
affected rows are data entry errors, not a real signal worth estimating.

```python
competitor_price_zeros_before = (df["competitorPrice"] == 0).sum()
df.loc[df["competitorPrice"] == 0, "competitorPrice"] = np.nan
```

### Difference

Estimating what the competitor price should have been for a data entry
error introduces invented data. There is no principled reason why the
bin-level average applies to these specific 976 rows. Recoding to NaN is
the simpler and more transparent decision: we do not know the value, so we
say we do not know it. The binary flag from Section 1 already captures that
competitor data was unavailable for these rows.

---

## 3. campaignIndex Encoding

**`scripts/utils.py`** one-hot encodes campaignIndex into three binary
columns (A, B, C). Rows with NaN get 0 in all three columns, which
implicitly represents "no campaign" without a named category.

```python
campaign_dummies = pd.get_dummies(df["campaignIndex"], prefix="campaignIndex")
df = pd.concat([df, campaign_dummies], axis=1)
for col in ["campaignIndex_A", "campaignIndex_B", "campaignIndex_C"]:
    if col not in df.columns:
        df[col] = 0
df = df.drop(columns=["campaignIndex"])
```

### My approach

I explicitly recoded NaN to the string "none" before encoding. This gives
the no-campaign state a named category rather than relying on an implicit
all-zeros representation. I also used `drop_first=True` which removes one
dummy column to avoid the dummy variable trap (where one column is perfectly
predictable from the others).

```python
campaign_missing_before = df["campaignIndex"].isna().sum()
df["campaignIndex"] = df["campaignIndex"].fillna("none")
dummies = pd.get_dummies(df[ONE_HOT_FEATURES], prefix=ONE_HOT_FEATURES, drop_first=True, dtype=int)
```

### Difference

Both approaches are functionally similar since 0,0,0 and a "none" dummy
column encode the same information. My preference for the explicit "none"
category is about readability and intention. When someone reads the feature
list and sees `campaignIndex_none`, they immediately understand what it
means. The implicit 0,0,0 encoding requires knowing that the absence of all
three flags means no campaign, which is not obvious.

Your approach also keeps all three campaign columns including campaignIndex_A,
which I dropped using `drop_first=True` to avoid multicollinearity. Kinda
the same idea with a different approach. Both are valid, but I prefer the
explicit "none" category for clarity and the dropped column for cleanliness.

---

## 4. content Column Parsing

**`scripts/utils.py`** parses content to integers, handles the X multiplier
format, and explicitly returns `pd.NA` for PAK and L 125. It also drops
rows where content is NaN after parsing.

```python
def parse_content(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    if "X" in value:
        parts = value.split("X")
        result = 1
        for part in parts:
            result *= int(float(part.strip()))
        return result
    elif 'PAK' in value:
        return pd.NA
    elif 'L   125' in value:
        return pd.NA
    return int(float(value))


new_df["content"] = new_df["content"].apply(parse_content)
new_df = new_df[new_df["content"].notna()]
```

### My approach

I separated the cleaning step from the parsing step, parsed to float instead
of int, added a regex classifier to understand the format distribution first,
and kept unparseable rows as NaN rather than dropping them.

```python
import re


def classify_content(value: str) -> str:
    if pd.isna(value):
        return 'missing'
    value = str(value).strip()
    if re.match(r'^\d+(\.\d+)?$', value):
        return 'numeric'
    if re.match(r'^\d+(\.\d+)?([Xx]\d+(\.\d+)?)+$', value):
        return 'multiplier'
    return 'unknown'


def parse_content(value):
    if pd.isna(value):
        return np.nan
    value_string = str(value).strip().upper()
    if "X" in value_string:
        parts = value_string.split("X")
        try:
            return float(parts[0]) * float(parts[1])
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(value_string)
    except ValueError:
        return np.nan


df["content_parsed"] = df["content"].apply(parse_content)
```

### Difference

**Float vs int:** Package sizes include decimals like 0.25, 0.6, and
30X0.6. Your int conversion truncates 30X0.6 to 18 instead of 18.0, which
is a minor but unnecessary rounding that loses real information.

**Dropping rows vs keeping NaN:** You dropped all rows where content cannot
be parsed. Only 2 products (PAK and L 125) are unparseable, appearing in
209 rows out of 2.75 million. Dropping 209 rows is effectively harmless, but
I prefer to keep them as NaN since those rows still have 14 other valid
features. Tree models handle NaN natively, as I mentioned before.

**Regex classifier:** I added a classification step before parsing to
understand the format distribution: 94.13% numeric, 5.86% multiplier, 0.01%
unknown. This is purely exploratory, it helped me understand what I was
dealing with before writing the parser.

---

## 5. pharmForm Cleaning

**`scripts/utils.py`** strips whitespace and uppercases the values only.
Missing values are returned unchanged as NaN.

```python
def normalize_pharmform(value):
    if pd.isna(value):
        return value
    return str(value).strip().upper()
```

### My approach

I recoded NaN to the explicit string "unknown" before encoding.

```python
pharmform_missing_before = df["pharmForm"].isna().sum()
df["pharmForm"] = df["pharmForm"].fillna("unknown")
```

### Difference

10.6% of products have no recorded dosage form. In a pharmaceutical
dataset, this is not random noise: products without a dosage form tend to
be supplements, devices, cosmetics, or food products that are not
traditional medications. This absence is informative and the model should
learn that "unknown dosage form" products behave differently from tablets
or capsules.

Leaving it as NaN with no explicit handling relies on the encoder to deal
with missingness implicitly and may not capture the specific signal that
"unknown" represents a distinct category of product rather than just a gap
in the data.

---

## 6. Price Feature Engineering

**`scripts/utils.py`** creates three price comparison features relative to
competitor price only.

```python
df["price_diff_competitor"] = (df["price"] - df["competitorPrice"]).round(2)
df["price_ratio_competitor"] = (df["price"] / df["competitorPrice"]).round(2)
df["price_pct_diff_competitor"] = (
        ((df["price"] - df["competitorPrice"]) / df["competitorPrice"]) * 100
).round(2)
```

### My approach

I created ratio features for both competitor price and RRP, plus a
log-transformed price. Three features but each capturing a distinct signal.

```python
df["price_to_rrp_ratio"] = df["price"] / df["rrp"]
df["price_to_competitor_ratio"] = df["price"] / df["competitorPrice"]
df["log_price"] = np.log(df["price"])
```

### Difference

**price_diff_competitor vs price_to_competitor_ratio:** The absolute
difference conflates price level with competitiveness. A difference of EUR 2
on a EUR 5 product means we are 40% cheaper. The same EUR 2 difference on a
EUR 100 product means we are 2% cheaper. The ratio captures relative
competitiveness regardless of price level, which is what actually matters for
consumer behavior.

**price_pct_diff_competitor:** Your percentage difference and my ratio
convey the same information scaled differently. I chose the ratio because it
is simpler and the log-odds interpretation in logistic regression is cleaner
with multiplicative features. I found this from online sources when reading
about feature engineering for pricing models.

**price_to_rrp_ratio:** This is the key addition you do not have. EDA showed
that price relative to the recommended retail price (RRP) is one of the top
predictors of purchase behavior. Customers respond to perceived discount
depth, not just the absolute price or the competitor comparison. A product
at 70% of RRP sends a very different signal than one at 99% of RRP, even if
both are cheaper than the competitor. This feature ended up in the top 5 of
every model I tested.

**log_price:** All three raw price columns are right-skewed with long tails
extending to EUR 379. Log transformation compresses this range and makes the
relationship with conversion more linear. The price elasticity analysis in
EDA confirmed the log relationship is appropriate since we fit logistic
regression on log(price) and got a statistically significant coefficient.

---

## 7. High-Cardinality Feature Handling

**`create_unified_dataset.py`** does not explicitly address high-cardinality
features. manufacturer, group, and category are left as-is after the merge.

### My approach

I used a data-driven frequency threshold: for each feature, find the minimum
count where values above it collectively cover 95% of all interactions.
Everything below that threshold becomes "other."

```python
COVERAGE_TARGET = 0.95


def find_frequency_threshold(series, coverage_target):
    counts = series.value_counts(dropna=False).sort_values(ascending=False)
    total = counts.sum()
    cumulative_coverage = counts.cumsum() / total
    values_needed = cumulative_coverage[cumulative_coverage <= coverage_target]
    if len(values_needed) == 0:
        return counts.iloc[0]
    threshold = counts.iloc[len(values_needed)]
    return threshold


for feature in HIGH_CARDINALITY_FEATURES:
    threshold = find_frequency_threshold(df[feature], COVERAGE_TARGET)
    rare_values = df[feature].value_counts()[lambda x: x < threshold].index
    df.loc[df[feature].isin(rare_values), feature] = "other"
```

### Difference

manufacturer has 1,065 unique values, group has 533, category has 409. The
bottom 50% of manufacturer values cover only 1% of interactions. A model
cannot learn anything reliable from a category it has seen once or twice, it
just adds noise. Grouping rare values into "other" keeps the signal from
common values while discarding the noise from rare ones.

The 95% threshold is data-driven rather than arbitrary. pharmForm needed a
threshold of 3,904 interactions to reach 95%, while manufacturer only needed

959. A single fixed number would over-group one feature and under-group
     another.

---

## 8. Encoding Strategy

**`create_unified_dataset.py`** explicitly encodes campaignIndex only. Other
categorical features are not addressed in the provided files.

### My approach

I split encoding into two strategies based on cardinality. Low-cardinality
features (4 values each) get one-hot encoding. High-cardinality features get
label encoding. CatBoost gets its own separate dataset with raw string
categoricals.

```python
ONE_HOT_FEATURES = ["availability", "salesIndex", "campaignIndex"]
dummies = pd.get_dummies(df[ONE_HOT_FEATURES], prefix=ONE_HOT_FEATURES, drop_first=True, dtype=int)
df = pd.concat([df, dummies], axis=1)
df = df.drop(columns=ONE_HOT_FEATURES)

LABEL_ENCODE_FEATURES = ["pharmForm", "manufacturer", "group", "category"]
label_encoding_maps = {}
for feature in LABEL_ENCODE_FEATURES:
    codes, uniques = pd.factorize(df[feature])
    df[feature] = codes
    label_encoding_maps[feature] = dict(enumerate(uniques))
```

For CatBoost specifically, I saved a separate version of the data where all
7 categorical columns remain as raw strings, because CatBoost builds its own
internal encoding using ordered target statistics and performs better when
it receives the original string values.

```python
CATBOOST_CATEGORICAL_COLUMNS = [
    "availability", "salesIndex", "campaignIndex",
    "pharmForm", "manufacturer", "group", "category",
]
catboost_df = df.drop(columns=CATBOOST_COLUMNS_TO_DROP).copy()
for col in CATBOOST_CATEGORICAL_COLUMNS:
    catboost_df[col] = catboost_df[col].astype(str)
```

### Difference

One-hot encoding 79 to 289 values per feature would add hundreds of sparse
binary columns. For tree-based models this adds memory and training time with
no benefit since trees split on threshold values and handle integer codes
natively. The CatBoost exception turned out to be the decisive factor: giving
it raw string categoricals rather than label-encoded integers produced the
best results across all five models I tested.

---

## 9. Temporal Feature Engineering

**`sample_down.py`** derives a week column from the day column for
stratification purposes only. No temporal feature is passed to the model.

```python
sampled_df["week"] = (sampled_df["day"] // 7).astype(int)
sampled_df["strata"] = sampled_df["order"].astype(str) + "_" + sampled_df["week"].astype(str)
```

### My approach

I engineered a binary feature that directly captures the structural
behavioral break identified in EDA at day 26.

```python
SHIFT_DAY = 26
df["is_post_shift"] = (df["day"] >= SHIFT_DAY).astype(int)
```

### Difference

EDA showed a clear permanent structural change at day 26: daily interactions
jumped 59.8% while the order rate dropped 14.15 percentage points and never
recovered. The same product at the same price converted very differently
before and after this point.

Your approach preserves the temporal distribution in the sample, which is
good thinking. But it does not give the model direct information about which
behavioral regime an interaction belongs to. My `is_post_shift` feature tells
the model explicitly: this interaction happened in the high-volume,
low-conversion period. It appeared in the top 5 features of every model I
tested, confirming it carries real predictive signal.

---

## 10. Sampling Strategy

**`sample_down.py`** creates a 200k sample stratified by both order flag
and week, with a fallback to order-only stratification.

```python
sampled_df["week"] = (sampled_df["day"] // 7).astype(int)
sampled_df["strata"] = sampled_df["order"].astype(str) + "_" + sampled_df["week"].astype(str)
sampled_df, _ = train_test_split(
    sampled_df_valid,
    train_size=200_000,
    stratify=sampled_df_valid["strata"],
    random_state=random_state
)
```

### My approach

I used the full 2.75 million rows for all training and evaluation with a
simple 80/20 stratified train/test split.

```python
features_train, features_test, target_train, target_test = train_test_split(
    features,
    target,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=target
)
```

### Difference

Sampling to 200k from 2.75 million discards 93% of the data. With gradient
boosting models, more data almost always produces better models. I addressed
the temporal distribution concern differently through the `is_post_shift`
feature, which gives the model direct information about behavioral regimes
without throwing away data.

GPU acceleration made this practical: XGBoost trained in 1.7 seconds on
the full data training rows on CUDA. That said, your approach is completely
valid if faster iteration is the priority or if GPU is not available. I
personally prefer using all the data and letting the model decide what is
important.

---

## Overall

Your approach is primarily focused on imputation: fill every gap with an
estimate before modeling. My approach is primarily focused on information
preservation: keep gaps as NaN where possible, add binary flags to capture
missingness as signal, and let the model handle uncertainty natively.

There is no single right way to handle missing data. The best approach
depends on the context, the models being used, and the assumptions you are
willing to make. Your approach is more traditional and is the right choice
when working with models that do not handle NaN natively, such as logistic
regression or SVM.

The only model I trained that cannot handle NaN was logistic regression,
and I handled that case specifically with median imputation applied only
during the modeling phase, not during data preparation. That way the
imputation decision is tied to the model that needs it rather than baked
into the dataset for all models.

---

I hope that explains my thinking :-) and as I said, there is no right or
wrong here, just different approaches with different trade-offs. The key is to be intentional and data-driven in your
decisions.

That said, I do think my approach is more robust and better aligned with the capabilities of modern tree-based models,
which is why I ended up with better
results. But your approach is still valid and would work well with the right models and assumptions. The important thing
is to understand the implications of each decision and how it interacts with the modeling choices.

That was my personal thinking and at the end of the day, the decision is a team decision :-)