### Agas is a small Python library for pairing similar (or dissimilar) data series.

### Data Series Similarity
Agas defines similarity as the absolute difference between pairs of output
values from an aggregation function applied to the input series.
The default behavior of Agas is to maximize similarity on a single dimension
(e.g., means of the series in the input matrix) while minimizing
similarity on another dimension (e.g., the variance of the series).

### Motivation
The main motivation for this library is to provide a data description tool for
depicting time-series. It is customary to plot pairs of time series, where the
pair is composed of data which is similar on one dimension (e.g., mean value) but
dissmilar on another dimension (e.g., standard deviation).

### Setup
```pip install agas``` should work. Conda package coming soon!

### Usage 

`agas` exposes two main functions to the user 

1. `agas.pair_from_numpy`
This function e

2. `agas.pair_from_pandas`

### Documentation 
See [Here](github.io/agas)


### Contributing to 


### Misc.
The library name Agas is abbreviation for aggregated-series. Also, 'Agas' is
Hebrew for 'Pear'.