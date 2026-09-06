import sys
sys.path.insert(0, r'D:\Claude\projects\3号交易员')
from tests.test_fix_disclosure_cninfo import _cninfo_df, _COLS_CNINFO
from trader3.v2.disclosure_sync import _CNINFO_COL_MAP

print('_COLS_CNINFO:', _COLS_CNINFO)
print('_CNINFO_COL_MAP:', _CNINFO_COL_MAP)

df = _cninfo_df()
print('test df columns:', df.columns.tolist())
print(df)