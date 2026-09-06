@echo off
cd /d D:\Claude\projects\3号交易员
python -c "from trader3.v2.disclosure_sync import _BACKUP; print('_BACKUP:', repr(_BACKUP))"