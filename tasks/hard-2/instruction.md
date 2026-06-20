# Constrained Shortest Path

Implement a solution to the following problem in `/app/solve.py`:

Given a weighted directed graph and a budget constraint, find the shortest path from source to destination that costs **at most** the given budget (where "cost" and "distance" are two separate edge attributes).

**Input format** (read from `/app/input.json`):
```json
{
  "nodes": <int>,
  "edges": [[u, v, distance, cost], ...],
  "source": <int>,
  "destination": <int>,
  "budget": <int>
}
```

**Output**: write to `/app/output.json`:
```json
{
  "min_distance": <int or null if no path exists within budget>,
  "path": [<node_ids in order>] or null
}
```

**Constraints**:
- Up to 100 nodes, up to 500 edges
- All distances and costs are positive integers
- Must find the **minimum distance** path where total cost ≤ budget
- If multiple paths have the same minimum distance, return any one

Your solution must handle the case where no valid path exists (output `null` for both fields).

Run with: `python /app/solve.py`
