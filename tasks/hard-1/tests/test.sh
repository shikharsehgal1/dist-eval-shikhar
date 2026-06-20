#!/bin/bash
set -e
SCORE=0

python3 /app/pipeline.py 2>/dev/null

if [ ! -f /app/results.json ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Validate JSON parses
python3 -c "import json; json.load(open('/app/results.json'))" 2>/dev/null && SCORE=$((SCORE + 10))

# Check correct key name grand_total_revenue
python3 -c "
import json
d = json.load(open('/app/results.json'))
assert 'grand_total_revenue' in d, f'Missing grand_total_revenue key, got: {list(d.keys())}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 15))

# Check North region (orders: 1001=1250.50, 1003=2100.75, 1008=945.00, 1012=3100.00, 1016=675.25, 1020=1560.50 = 6 orders, total=9632.00)
python3 -c "
import json
d = json.load(open('/app/results.json'))
north = d['regions']['North']
assert north['num_orders'] == 6, f'North num_orders wrong: {north[\"num_orders\"]}'
assert north['total_revenue'] == 9632.0, f'North total wrong: {north[\"total_revenue\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check South region (1002=875, 1006=1100.50, 1010=1800, 1018=2300.75 = 4 valid, skip missing 1014)
python3 -c "
import json
d = json.load(open('/app/results.json'))
south = d['regions']['South']
assert south['num_orders'] == 4, f'South num_orders wrong: {south[\"num_orders\"]}'
assert south['total_revenue'] == 6076.25, f'South total wrong: {south[\"total_revenue\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check East region (1004=650.25, 1011=525.75, 1015=1450, 1019=780 = 4 valid, skip missing 1007)
python3 -c "
import json
d = json.load(open('/app/results.json'))
east = d['regions']['East']
assert east['num_orders'] == 4, f'East num_orders wrong: {east[\"num_orders\"]}'
assert east['total_revenue'] == 3406.0, f'East total wrong: {east[\"total_revenue\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check grand total (9632 + 6076.25 + 3406 + West)
python3 -c "
import json
d = json.load(open('/app/results.json'))
gt = d['grand_total_revenue']
assert abs(gt - 26280.25) < 0.01, f'grand_total_revenue wrong: {gt}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 15))

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
