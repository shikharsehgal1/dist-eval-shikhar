#!/usr/bin/env python3
"""
Sales data pipeline. Reads sales.csv, aggregates by region, writes results.json.
THIS FILE HAS BUGS - fix them without rewriting from scratch.
"""
import csv
import json


def load_sales(path):
    records = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Bug 1: revenue should be float(row['revenue']), not int()
            try:
                revenue = int(row['revenue'])
            except (ValueError, TypeError):
                revenue = 0  # Bug 2: missing values should be skipped, not zeroed
            
            region = row.get('region', '').strip()
            if not region:
                continue
            
            records.append({
                'region': region,
                'revenue': revenue,
                'order_id': row.get('order_id', ''),
            })
    return records


def aggregate(records):
    regions = {}
    for rec in records:
        r = rec['region']
        if r not in regions:
            regions[r] = {'total_revenue': 0, 'num_orders': 0}
        regions[r]['total_revenue'] += rec['revenue']
        regions[r]['num_orders'] += 1  # Bug 3: this counts all records, but missing-revenue rows were zeroed and included above instead of skipped

    result = {}
    for r, data in regions.items():
        avg = data['total_revenue'] / data['num_orders']
        result[r] = {
            'total_revenue': round(data['total_revenue'], 2),
            'avg_order_value': round(avg, 2),
            'num_orders': data['num_orders'],
        }
    return result


def main():
    records = load_sales('/app/sales.csv')
    agg = aggregate(records)
    grand_total = sum(r['total_revenue'] for r in agg.values())  # Bug 4: should sum from agg values, but agg values use rounded numbers which is fine; actual bug: grand_total key is wrong below
    
    output = {
        'regions': agg,
        'grand_total': round(grand_total, 2),  # Bug 4: key should be 'grand_total_revenue'
    }
    
    with open('/app/results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Pipeline complete. Processed {len(records)} records across {len(agg)} regions.")


if __name__ == '__main__':
    main()
