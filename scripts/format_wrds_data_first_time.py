from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import DataManager
import sys
import logging
import pandas as pd
pd.set_option("display.max_columns", None)

# In run_pipeline_config.json:
# "S3_PRICES_PATH": "s3://dynamic-allocation-macro-fmp/data/wrds_gross_query.parquet"

logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
config = Config()

data_manager = DataManager(config=config)
res = data_manager._fetch_data_from_s3()
df = res["prices"].copy()
df["custom_month"] = (
    df["date"].dt.to_period("M")
    - (df["date"].dt.day <= 2).astype(int)
)
monthly_returns = (
    df
    .groupby(["permno", "custom_month"])["ret"]
    .apply(lambda x: (1 + x).prod() - 1)
    .reset_index()
)
monthly_returns["date"] = (
    monthly_returns["custom_month"]
    .dt.to_timestamp()
    + pd.offsets.MonthBegin(1)
    + pd.offsets.Day(1)
)
prices = monthly_returns.pivot(columns="permno", index="date", values="ret")
prices.to_parquet(config.ROOT_DIR / "data" / "market" / "monthly_ret.parquet")


