#!/bin/bash
cat > /app/analyze.py << 'EOF'
import json
from collections import defaultdict

counts = defaultdict(int)
hours = defaultdict(int)
total = 0
kw_timeout = 0
kw_failed = 0

with open('/app/server.log') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total += 1
        parts = line.split(' ', 3)
        if len(parts) >= 3:
            level = parts[2]
            counts[level] += 1
            try:
                hour = int(parts[1].split(':')[0])
                hours[hour] += 1
            except Exception:
                pass
        if 'timeout' in line.lower():
            kw_timeout += 1
        if 'failed' in line.lower():
            kw_failed += 1

error_count = counts.get('ERROR', 0) + counts.get('CRITICAL', 0)
most_common_hour = max(hours, key=hours.get) if hours else 0

report = {
    'total_lines': total,
    'by_level': {lvl: counts.get(lvl, 0) for lvl in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']},
    'error_rate': round(error_count / total, 2) if total else 0.0,
    'most_common_hour': most_common_hour,
    'messages_with_keyword': {'timeout': kw_timeout, 'failed': kw_failed},
}
with open('/app/report.json', 'w') as f:
    json.dump(report, f, indent=2)
EOF
python3 /app/analyze.py
