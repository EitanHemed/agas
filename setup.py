"""
Agas is a small Python library for pairing similar (or dissimilar) data series.

Agas defines similarity as the absolute difference between pairs of output
scores from an aggregation function applied to the input series.
The default behavior of Agas is to maximize similarity on a single dimension
(e.g., means of the series in the input matrix) while minimizing
similarity on another dimension (e.g., the variance of the series).

The main motivation for this library is to provide a data description tool for
depicting time-series. It is customary to plot pairs of time series, where the
pair is composed of data which is similar on one dimension (e.g., mean value) but
dissmilar on another dimension (e.g., standard deviation).

The library name Agas is abbreviation for aggregated-series. Also, 'Agas' is
Hebrew for 'Pear'.
"""
DOCLINES = (__doc__ or '').split("\n")

import setuptools

VERSION = "0.0.1"

INSTALL_REQUIRES = [
    'numpy>=1.13.3',
    'pandas>=0.25',
]

EXTRAS_REQUIRE = {}

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Operating System :: OS Independent',
]

setuptools.setup(
    name="agas",
    version=VERSION,
    author="Eitan Hemed",
    author_email="Eitan.Hemed@gmail.com",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    url="https://github.com/EitanHemed/agas",
    project_urls={
        "Bug Tracker": "https://github.com/EitanHemed/agas/issues",
    },
    classifiers=CLASSIFIERS,
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifies=CLASSIFIERS,
)
