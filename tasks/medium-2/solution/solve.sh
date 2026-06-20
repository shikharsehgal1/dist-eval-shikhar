#!/bin/bash
# Start mock server
python3 /app/mock_server.py &
sleep 2

cat > /app/client.py << 'EOF'
import json
import urllib.request
from collections import defaultdict

with urllib.request.urlopen('http://localhost:5000/users') as r:
    users = json.loads(r.read())

eligible = [u for u in users if u['age'] >= 30]
depts = defaultdict(list)
for u in eligible:
    depts[u['department']].append(u['salary'])

departments = {}
for dept, salaries in depts.items():
    departments[dept] = {
        'count': len(salaries),
        'avg_salary': round(sum(salaries) / len(salaries), 2),
    }

summary = {'departments': departments, 'total_eligible_users': len(eligible)}
with open('/app/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
EOF
python3 /app/client.py
