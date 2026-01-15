from dynamic_allocation_macro_fmp.utils.config import Config
from dynamic_allocation_macro_fmp.data.data import CSVDataSource
config = Config()
data_src = CSVDataSource(
    file_path=config.fred_path
)