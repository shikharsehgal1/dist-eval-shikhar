# Log File Analyzer

A log file exists at `/app/server.log`. Each line has the format:
```
YYYY-MM-DD HH:MM:SS LEVEL message
```
where LEVEL is one of: DEBUG, INFO, WARNING, ERROR, CRITICAL

Write a Python script at `/app/analyze.py` that reads `/app/server.log` and writes a JSON report to `/app/report.json` with the following structure:
```json
{
  "total_lines": <int>,
  "by_level": {
    "DEBUG": <int>,
    "INFO": <int>,
    "WARNING": <int>,
    "ERROR": <int>,
    "CRITICAL": <int>
  },
  "error_rate": <float, 2 decimal places, fraction of lines that are ERROR or CRITICAL>,
  "most_common_hour": <int, 0-23, hour with most log entries>,
  "messages_with_keyword": {
    "timeout": <int, count of lines containing 'timeout' case-insensitive>,
    "failed": <int, count of lines containing 'failed' case-insensitive>
  }
}
```

The script must handle the existing `/app/server.log` file and produce valid JSON.
