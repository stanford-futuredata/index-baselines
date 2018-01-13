# Index Baselines

Simple baselines for "Learned Indexes" to complement the [blog post](http://dawn.cs.stanford.edu/2018/01/11/index-baselines/).

## Build

`make`

## Generating Data

`gen_lognormal` generates 190M unique 32-bit integers of the form int(1e+9 * x), where x is a sample from an (0, 2) log-normal distribution.

The integers are written to the binary file `lognormal.sorted.190M`.

## Hashing

To run our SIMD-enabled bucketized cuckoo hash table hashing baseline:

`sh run_hashing.sh lognormal.sorted.190M`

Our implementation uses 32-bit integer keys and values. On an Intel Xeon E5-2690v4 CPU (2.6GHz), our table took 36ns per access while wasting 0.015GB of space (1% of slots).

## Range Search

As with the hashing baselines, we use 32-bit integer keys and values. This task makes the additional assumption that the data is sorted by key.

To run our range search baselines:

`sh run_range_search.sh lognormal.sorted.190M`

This script runs several baselines:

- binary search
- [stx::btree](https://github.com/bingmann/stx-btree), an open-source B-Tree implementation
- a simple read-only B-Tree with a two-level index
- a simple read-only B-Tree with a three-level index
- [FAST](http://dl.acm.org/citation.cfm?id=1807206), a fast SIMD-enabled B-Tree implementation

Our simple read-only B-Trees perform binary search on the topmost level, followed by AVX2 linear search on the subsequent levels until the position of the queried key is found. We found that pages of size 48 work well on our hardware setup; the page size can be tuned for your own hardware by calling the `range_search` binary with different arguments.

On an Intel Xeon E5-2690v4 CPU (2.6GHz), we observe the following average query times on the log-normal data:

| method | query time (ns) |
| --- | --- |
| binary search | 785.7 |
| stx::btree | 534.1 | 
| 2-level index | 201.1 |
| 3-level index | 177.3 |
| FAST | 125.7 |

We note that directly comparing these numbers against those reported in the paper should be done with a grain of salt due to unspecified differences in testing environments.
In particular, CPU cache sizes have a significant impact on performance for this workload.
A second caveat is that the methods evaluated in the paper use binary search on the base data, whereas the simple B-Tree and FAST baselines use vectorized linear search.
While it's possible that the learned index approach could be accelerated with the use of SIMD-based linear search on the lowest level, this potential optimization was not evaluated in the paper.
