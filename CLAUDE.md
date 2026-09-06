import re

with open('CLAUDE.md', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Update test count
content = re.sub(r'tests/.*?(\d+)\s*个?测试.*?(\d+)\s*passed.*?(\d+)\s*skipped',
                 'tests/              # 601 个测试（599 passed / 2 skipped 基线）',
                 content)

with open('CLAUDE.md', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated')