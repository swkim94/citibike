# NYC Citi Bike Analysis

Analysis of ~10 million NYC Citi Bike trips (all of 2015, ~1.7GB across 12 monthly CSVs) — a Data Incubator take-home challenge.

## What it does

- Loads and concatenates 12 months of trip data into a single indexed pandas DataFrame
- Implements Welford's online (single-pass) algorithm for running mean/variance from scratch, and benchmarks it against pandas' built-in `.std()`/`.mean()`
- Computes great-circle distance between start/end stations from lat/long (haversine-style spherical distance)
- Cleans the data: removes same-station round trips, caps unreasonable trip durations, filters implausible speeds (>50 km/h)
- Answers a series of analysis questions, including:
  - Median trip duration
  - Fraction of rides starting/ending at the same station
  - Variance in the number of stations visited per bike
  - Average trip distance (using the running-average implementation)
  - Month with the largest swing in average trip duration
  - Peak hourly usage concentration across stations
  - Fraction of rides exceeding usage time limits
  - Average number of times a bike changes location

## Files

| File | Description |
|---|---|
| `Q2a.py` | Main analysis script — data loading, cleaning, running statistics, and all challenge questions |
| `Q2b.py` | Variant/continuation of the analysis |
| `SK_Map.png` | Station location scatter plot |

## Requirements

```
numpy
pandas
matplotlib
```

## Note

Trip data (`*-citibike-tripdata.csv`) is not included — source from [Citi Bike System Data](https://www.citibikenyc.com/system-data).
