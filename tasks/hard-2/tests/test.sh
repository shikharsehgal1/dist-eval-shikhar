#!/bin/bash
set -e
SCORE=0

python3 /app/solve.py 2>/dev/null

if [ ! -f /app/output.json ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Validate JSON parses
python3 -c "import json; json.load(open('/app/output.json'))" 2>/dev/null && SCORE=$((SCORE + 10))

# Check min_distance
# Budget=5: paths from 0 to 5:
# 0->5 direct: dist=20, cost=1 (within budget, but dist=20)
# 0->1->3->5: dist=4+3+2=9, cost=2+1+1=4 <= 5 ✓
# 0->2->1->3->5: dist=2+1+3+2=8, cost=5+2+1+1=9 > 5 ✗
# 0->1->4->5: dist=4+6+3=13, cost=2+3+2=7 > 5 ✗
# 0->2->4->5: dist=2+5+3=10, cost=5+6+2=13 > 5 ✗
# Best within budget: 0->1->3->5 dist=9 OR 0->5 dist=20
# Min distance = 9
python3 -c "
import json
d = json.load(open('/app/output.json'))
assert d['min_distance'] == 9, f'min_distance wrong: {d[\"min_distance\"]}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 40))

# Check path is valid and has correct distance/cost
python3 -c "
import json
d = json.load(open('/app/output.json'))
path = d['path']
assert path is not None, 'path is null'
assert path[0] == 0, 'path must start at 0'
assert path[-1] == 5, 'path must end at 5'
# Verify path distance = 9 using the graph
edges = {(0,1):(4,2),(0,2):(2,5),(1,3):(3,1),(1,4):(6,3),(2,1):(1,2),(2,3):(8,1),(2,4):(5,6),(3,5):(2,1),(4,5):(3,2),(0,5):(20,1)}
total_dist = 0
total_cost = 0
for i in range(len(path)-1):
    e = (path[i], path[i+1])
    assert e in edges, f'Invalid edge {e}'
    total_dist += edges[e][0]
    total_cost += edges[e][1]
assert total_dist == 9, f'Path dist wrong: {total_dist}'
assert total_cost <= 5, f'Path cost exceeds budget: {total_cost}'
print('ok')
" 2>/dev/null && SCORE=$((SCORE + 50))

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
