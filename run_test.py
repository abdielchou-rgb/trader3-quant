import sys
import os
sys.path.insert(0, r'D:\Claude\projects\3号交易员')
os.chdir(r'D:\Claude\projects\3号交易员')

import pytest
import sys

if __name__ == '__main__':
    sys.exit(pytest.main(['-xvs', 'tests/test_fix_disclosure_cninfo.py::test_primary_fails_cninfo_backup_succeeds', '-q', '--tb=short']))