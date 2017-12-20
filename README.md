# index-baselines
Simple baselines for "Learned Indexes"

## Build

Run `make`.

## Generating Data

`gen_lognormal` generates 190M unique integers of the form int(1e+9 * x), where x is a sample from an (0, 2) log-normal distribution.

## Hashing 

To run the hashing baseline:

`hashing <path>`

## Range Search

To run the range search baselines:

`sh run_range_search.sh <path>`
