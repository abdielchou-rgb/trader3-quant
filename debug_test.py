import sys
sys.path.insert(0, r'D:\Claude\projects\3号交易员')
from tests.test_fix_disclosure_cninfo import _cninfo_df
import pandas as pd

df = _cninfo_df()
print('columns:', df.columns.tolist())
print(df)