import pandas as pd


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    pct = 100 * miss / len(df)
    return pd.DataFrame({'missing_count': miss, 'missing_pct': pct})


def drop_high_missing(df: pd.DataFrame, thresh_pct: float = 50.0) -> pd.DataFrame:
    """Drop columns with more than `thresh_pct` missing."""
    to_drop = df.columns[df.isna().mean() * 100 > thresh_pct]
    return df.drop(columns=to_drop)


def fill_missing_numeric(
        df: pd.DataFrame,
        cols: list,
        strategy: str = 'median'
) -> pd.DataFrame:
    for c in cols:
        val = df[c].median() if strategy == 'median' else df[c].mean()
        df[c] = df[c].fillna(val)
    return df


def fill_missing_categorical(
        df: pd.DataFrame,
        cols: list,
        fill_value: str = 'Unknown'
) -> pd.DataFrame:
    for c in cols:
        df[c] = df[c].fillna(fill_value)
    return df


def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset)


def cap_outliers_iqr(
        df: pd.DataFrame,
        cols: list,
        factor: float = 1.5,
        positive_only: bool = False
) -> pd.DataFrame:
    for c in cols:
        if positive_only:
            mask = df[c] > 0
            q1, q3 = df.loc[mask, c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            df.loc[mask, c] = df.loc[mask, c].clip(lower, upper)
        else:
            q1, q3 = df[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
            df[c] = df[c].clip(lower, upper)
    return df