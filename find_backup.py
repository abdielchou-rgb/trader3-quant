import sys
with open('trader3/v2/disclosure_sync.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if '_BACKUP' in line:
            print(f'{i}: {line.rstrip()}')