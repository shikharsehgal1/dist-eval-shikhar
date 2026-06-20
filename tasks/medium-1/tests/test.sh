#!/bin/bash
set -e
SCORE=0

if [ ! -f /app/analyze.py ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

python3 /app/analyze.py 2>/dev/null

if [ ! -f /app/report.json ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Validate JSON parses
python3 -c "import json; json.load(open('/app/report.json'))" 2>/dev/null && SCORE=$((SCORE + 10))

# Check total_lines = 40
python3 -c "
import json
d = json.load(open('/app/report.json'))
assert d['total_lines'] == 40, f'total_lines wrong: {d[\"total_lines\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check by_level counts
python3 -c "
import json
d = json.load(open('/app/report.json'))
bl = d['by_level']
assert bl['ERROR'] == 9, f'ERROR count wrong: {bl[\"ERROR\"]}'
assert bl['WARNING'] == 6, f'WARNING count wrong: {bl[\"WARNING\"]}'
assert bl['CRITICAL'] == 2, f'CRITICAL count wrong: {bl[\"CRITICAL\"]}'
assert bl['INFO'] == 16, f'INFO count wrong: {bl[\"INFO\"]}'
assert bl['DEBUG'] == 7, f'DEBUG count wrong: {bl[\"DEBUG\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 30))

# Check error_rate (ERROR+CRITICAL = 11/40 = 0.28 rounded to 2 dp)
python3 -c "
import json
d = json.load(open('/app/report.json'))
er = round(d['error_rate'], 2)
assert er == 0.28, f'error_rate wrong: {er}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check keyword counts
python3 -c "
import json
d = json.load(open('/app/report.json'))
kw = d['messages_with_keyword']
assert kw['timeout'] == 4, f'timeout count wrong: {kw[\"timeout\"]}'
assert kw['failed'] == 3, f'failed count wrong: {kw[\"failed\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
