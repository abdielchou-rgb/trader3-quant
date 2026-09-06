import sys
sys.path.insert(0, r'D:\Claude\projects\3号交易员')
from tests.test_fix_disclosure_cninfo import _cninfo_df
from trader3.v2.disclosure_sync import _CNINFO_COL_MAP
print('_CNINFO_COL_MAP:', _CNINFO_COL_MAP)
df = __import__('tests.test_fix_disclosure_cninfo', fromlist=['_cninfo_df'])._cninfo_df()
print(df.columns.tolist())