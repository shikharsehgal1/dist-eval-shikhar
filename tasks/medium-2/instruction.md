# REST API Client

A mock API server is running at `http://localhost:5000`. It exposes:

- `GET /users` — returns a JSON array of user objects, each with fields: `id` (int), `name` (str), `age` (int), `department` (str), `salary` (float)
- `GET /users/<id>` — returns a single user by ID

Write a Python script at `/app/client.py` that:

1. Fetches all users from `GET /users`
2. Filters to users aged 30 or older
3. Groups them by department
4. For each department, computes the average salary (rounded to 2 decimal places)
5. Writes results to `/app/summary.json` in this format:
```json
{
  "departments": {
    "<dept_name>": {
      "count": <int>,
      "avg_salary": <float>
    }
  },
  "total_eligible_users": <int>
}
```

The script should run with `python /app/client.py` and produce `/app/summary.json`.
