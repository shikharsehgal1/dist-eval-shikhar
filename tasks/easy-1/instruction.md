# Word Count Script

Write a Python script at `/app/wordcount.py` that:

1. Reads text from stdin
2. Counts the total number of words (whitespace-separated tokens)
3. Counts the total number of lines
4. Counts the total number of unique words (case-insensitive)
5. Prints output in exactly this format:
```
words: <N>
lines: <N>
unique: <N>
```

Example: given input `Hello world\nhello again`, output should be:
```
words: 4
lines: 2
unique: 3
```

For empty input (e.g. a single empty line), report `words: 0`, `lines: 1`, `unique: 0`.
