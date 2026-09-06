@echo off
cd /d D:\Claude\projects\3号交易员
python -c "import sys; sys.path.insert(0, '.'); from tests.test_fix_disclosure_cninfo import _cninfo_df, _CNINFO_COL_MAP; import pandas as pd; df = _cninfo_df(); print('map:', _CNINFO_COL_MAP); print('cols:', df.columns.tolist()); print(df)"