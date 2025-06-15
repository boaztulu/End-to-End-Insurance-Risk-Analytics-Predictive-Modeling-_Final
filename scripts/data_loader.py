# scripts/data_loader.py
import pandas as pd
from typing import List, Dict, Optional

def load_insurance_data(
    filepath: str,
    sep: str = '\t',                           # default to tab-delimited
    dtypes: Optional[Dict[str, object]] = None,
    parse_dates: Optional[List[str]] = None
) -> pd.DataFrame:

    df = pd.read_csv(
        filepath,
        sep=sep,
        dtype=dtypes,
        parse_dates=parse_dates,
        low_memory=False
    )
    return df

def load_clean_data(
  filepath: str
)-> pd.DataFrame:
    df_clean = pd.read_csv(filepath)
    return df_clean