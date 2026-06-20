#!/bin/bash
cat > /app/wordcount.py << 'EOF'
import sys

text = sys.stdin.read()
lines = text.splitlines()
words = text.split()
unique = set(w.lower() for w in words)
print(f"words: {len(words)}")
print(f"lines: {len(lines)}")
print(f"unique: {len(unique)}")
EOF
