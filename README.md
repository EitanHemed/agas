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

`agas` exposes the following functions:

#### 1. `agas.pair_from_numpy`
By default, `agas.pair_from_numpy` will return the indices of the pair of arrays
which maximize similarity on the function supplied as the `maximize` argument
while minimizing similarity on the function supplied as the `minimize` argument. 
```
import numpy as np
import agas

a = np.vstack([[0, 0.5], [0.5, 0.5], [5, 5], [4, 10]])

agas.pair_from_array(a, maximize_function=np.std, minimize_function=np.mean)
# Output: (0, 2) 
```

`a[(0, 2), :]` is `[[0, 0.5], [5, 5]]` which provide a mixture of similar variance
and a large difference in means. We can prioritize similarity in variance over 
dissimilarity in mean value using the `maximize_weight` (defaults to `0.5`)
which the weights input from the maximizing-similarity function (here `np.std`) 
vs. the minimizing-similarity function.

`maximize_weight` can be set between 0 and 1 (inclusive). 
Automatically, `minimize_weight` will be `1 - maximize_weight`.
```
# Continued with the array from above
agas.pair_from_array(a, maximize_function=np.std, minimize_function=np.mean, 
    maximize_weight=.7)
# Output: (1, 2)
```


#### 2. `agas.pair_from_pandas`
#### todo - EXAMPLE 


### Documentation 
See [Here](github.io/EitanHemed/agas).


### Bug reports
Please open an [issue](https://github.com/EitanHemed/agas/issues). 


### Misc.
The library name Agas is abbreviation for aggregated-series. Also, 'Agas' is
Hebrew for 'Pear'.