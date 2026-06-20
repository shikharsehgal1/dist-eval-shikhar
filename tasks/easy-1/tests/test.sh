#!/bin/bash
set -e
SCORE=0

# Check script exists
if [ ! -f /app/wordcount.py ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Test 1: basic word count
RESULT=$(echo "Hello world" | python3 /app/wordcount.py 2>/dev/null)
if echo "$RESULT" | grep -q "words: 2" && echo "$RESULT" | grep -q "lines: 1" && echo "$RESULT" | grep -q "unique: 2"; then
    SCORE=$((SCORE + 34))
fi

# Test 2: multiline with repeats
RESULT=$(printf "Hello world\nhello again" | python3 /app/wordcount.py 2>/dev/null)
if echo "$RESULT" | grep -q "words: 4" && echo "$RESULT" | grep -q "lines: 2" && echo "$RESULT" | grep -q "unique: 3"; then
    SCORE=$((SCORE + 33))
fi

# Test 3: empty input
RESULT=$(echo "" | python3 /app/wordcount.py 2>/dev/null)
if echo "$RESULT" | grep -q "words: 0" && echo "$RESULT" | grep -q "lines: 1"; then
    SCORE=$((SCORE + 33))
fi

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
