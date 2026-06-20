#!/bin/bash
set -e
SCORE=0

if [ ! -f /app/fizzbuzz.py ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Run fizzbuzz for N=15
python3 /app/fizzbuzz.py 15 2>/dev/null

if [ ! -f /app/output.txt ]; then
    echo "0" > /logs/verifier/reward.txt
    exit 0
fi

# Check line 3 is Fizz
LINE3=$(sed -n '3p' /app/output.txt)
[ "$LINE3" = "Fizz" ] && SCORE=$((SCORE + 25))

# Check line 5 is Buzz
LINE5=$(sed -n '5p' /app/output.txt)
[ "$LINE5" = "Buzz" ] && SCORE=$((SCORE + 25))

# Check line 15 is FizzBuzz
LINE15=$(sed -n '15p' /app/output.txt)
[ "$LINE15" = "FizzBuzz" ] && SCORE=$((SCORE + 25))

# Check total line count is 15
LINECOUNT=$(wc -l < /app/output.txt | tr -d ' ')
[ "$LINECOUNT" = "15" ] && SCORE=$((SCORE + 25))

python3 -c "print($SCORE / 100.0)" > /logs/verifier/reward.txt
