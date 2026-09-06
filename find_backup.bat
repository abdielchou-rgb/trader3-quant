@echo off
cd /d D:\Claude\projects\3号交易员
python -c "with open('trader3/v2/disclosure_sync.py', 'r', encoding='utf-8') as f: lines = f.readlines(); [print(f'{i}: {line.rstrip()}') for i, line in enumerate(lines, 1) if '_BACKUP' in line]"