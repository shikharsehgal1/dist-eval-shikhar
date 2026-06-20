# FizzBuzz

Write a Python script at `/app/fizzbuzz.py` that:

1. Takes a single integer argument N from the command line
2. For each number from 1 to N (inclusive):
   - Prints "FizzBuzz" if divisible by both 3 and 5
   - Prints "Fizz" if divisible by 3 only
   - Prints "Buzz" if divisible by 5 only
   - Otherwise prints the number
3. Writes all output to `/app/output.txt` (one entry per line)

Example: `python /app/fizzbuzz.py 15` should produce `/app/output.txt` containing:
```
1
2
Fizz
4
Buzz
Fizz
7
8
Fizz
Buzz
11
Fizz
13
14
FizzBuzz
```
