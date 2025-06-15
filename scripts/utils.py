import pandas as pd

def summarize_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Returns mean, std, min, max for a list of numeric columns.
    """
    return df[cols].describe().T[['mean','std','min','max']]

def loss_ratio(df: pd.DataFrame) -> float:
    """
    Overall loss ratio = TotalClaims / TotalPremium
    """
    return df['TotalClaims'].sum() / df['TotalPremium'].sum()
