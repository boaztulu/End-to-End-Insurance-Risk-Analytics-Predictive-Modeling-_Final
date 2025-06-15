import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.proportion import proportions_ztest

def add_claim_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - has_claim: 1 if TotalClaims > 0, else 0
      - margin: TotalPremium - TotalClaims
    """
    df = df.copy()
    df['has_claim'] = (df['TotalClaims'] > 0).astype(int)
    df['margin']    = df['TotalPremium'] - df['TotalClaims']
    return df

def test_chi2_claim_frequency(df: pd.DataFrame, group_col: str):
    """
    Chi-square test for differences in claim frequency (has_claim)
    across categories in group_col.
    Returns (chi2, p_value, dof, expected_freq_table).
    """
    ct = pd.crosstab(df[group_col], df['has_claim'])
    chi2, p, dof, expected = chi2_contingency(ct)
    return chi2, p, dof, expected

def test_anova_margin(df: pd.DataFrame, group_col: str):
    """
    One-way ANOVA test for differences in margin across group_col.
    Returns (F_statistic, p_value).
    """
    groups = [grp['margin'].values for _, grp in df.groupby(group_col)]
    f_stat, p = f_oneway(*groups)
    return f_stat, p

def test_proportion_z(df: pd.DataFrame, group_col: str, grpA, grpB):
    """
    Z-test for difference in claim frequency between two groups grpA vs. grpB.
    Returns (z_stat, p_value).
    """
    sub = df[df[group_col].isin([grpA, grpB])]
    counts = sub.groupby(group_col)['has_claim'].sum().astype(int)
    nobs   = sub.groupby(group_col)['has_claim'].count().astype(int)
    z_stat, p = proportions_ztest(counts.values, nobs.values)
    return z_stat, p
