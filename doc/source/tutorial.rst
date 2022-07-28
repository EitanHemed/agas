:notoc:
.. _tutorial:

=======================
User guide and tutorial
=======================

************
Step-by-step
************

Finding the optimal pair is composed of several steps:

Agas is applied to `A` a matrix of size N X T where N is the number of nesting
units (countries, sensors, etc.) and T is the number of samples nested
within each unit (timestamps).

1. Each row in the data frame is aggregated using both MAXIMIZE
and MINIMIZE, producing two vectors of summary statistics
(e.g., means).

2. For each of the vectors, the normalized differences (between 0 and 1) are
calculated for any element, relative to each of the other elements on that
vector.

3.