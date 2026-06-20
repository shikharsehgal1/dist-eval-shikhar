#!/bin/bash
set -e
SCORE=0

# Start mock server
python3 /app/mock_server.py &
SERVER_PID=$!
sleep 2

if [ ! -f /app/client.py ]; then
    kill $SERVER_PID 2>/dev/null
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

python3 /app/client.py 2>/dev/null

kill $SERVER_PID 2>/dev/null

if [ ! -f /app/summary.json ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Validate JSON
python3 -c "import json; json.load(open('/app/summary.json'))" 2>/dev/null && SCORE=$((SCORE + 10))

# Check total_eligible_users (age >= 30: Alice 34, Carol 41, Eve 35, Frank 52, Henry 38, Iris 31, Jack 45 = 7)
python3 -c "
import json
d = json.load(open('/app/summary.json'))
assert d['total_eligible_users'] == 7, f'total_eligible_users wrong: {d[\"total_eligible_users\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 25))

# Check Engineering dept (Alice 95000, Carol 110000, Frank 130000 -> avg 111666.67)
python3 -c "
import json
d = json.load(open('/app/summary.json'))
eng = d['departments']['Engineering']
assert eng['count'] == 3, f'Engineering count wrong: {eng[\"count\"]}'
assert eng['avg_salary'] == 111666.67, f'Engineering avg wrong: {eng[\"avg_salary\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 25))

# Check Sales dept (Jack 85000 only, age>=30 -> avg 85000.0, count 1)
python3 -c "
import json
d = json.load(open('/app/summary.json'))
sales = d['departments']['Sales']
assert sales['count'] == 1, f'Sales count wrong: {sales[\"count\"]}'
assert sales['avg_salary'] == 85000.0, f'Sales avg wrong: {sales[\"avg_salary\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

# Check HR dept (Henry 75000, Iris 68000 -> avg 71500.0, count 2)
python3 -c "
import json
d = json.load(open('/app/summary.json'))
hr = d['departments']['HR']
assert hr['count'] == 2, f'HR count wrong: {hr[\"count\"]}'
assert hr['avg_salary'] == 71500.0, f'HR avg wrong: {hr[\"avg_salary\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 20))

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
