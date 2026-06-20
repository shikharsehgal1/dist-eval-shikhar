# Debug a Broken Data Pipeline

The file `/app/pipeline.py` is a data processing pipeline that is supposed to:
1. Read a CSV file of sales records from `/app/sales.csv`
2. Clean the data (handle missing values, fix data types)
3. Compute per-region aggregates: total revenue, average order value, number of orders
4. Write results to `/app/results.json`

However, the pipeline has **multiple bugs** that cause it to either crash or produce wrong results. Your job is to:

1. Understand what the pipeline is supposed to do by reading the code
2. Identify all the bugs
3. Fix them so the pipeline runs correctly and produces valid output

The correct `/app/results.json` must have this structure:
```json
{
  "regions": {
    "<region_name>": {
      "total_revenue": <float, rounded to 2 decimal places>,
      "avg_order_value": <float, rounded to 2 decimal places>,
      "num_orders": <int>
    }
  },
  "grand_total_revenue": <float, rounded to 2 decimal places>
}
```

Do NOT rewrite the pipeline from scratch — fix the existing code.
