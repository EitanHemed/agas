:notoc:
.. _tutorial:

=======================
User guide and tutorial
=======================

************
Step-by-step
************

Finding the optimal pair is composed of several steps:

1. Take `A` a matrix of size N X T where N is the number of nesting
units (countries, sensors, etc.) and T is the number of samples nested
within each unit (timestamps).

2. Generate two vectors of summary statistics one for each of the `maximize`
and `minimize` functions. Each vector is the output of applying a function on
each of the rows in `A`.

2. For each of the vectors, construct a matrix of the absolute differences
between any pair of its elements. Scale the differences by normalizing the
differences (between 0 and 1).

3. Calculate a weighted average of the two normalized matrices by
summing the matrices when multiplied by weight assigned to each of the `maximize`
and `minimize` functions.

4. Find the element in the matrix with the smallest value.

.. math::
   (a + b)^2 = a^2 + 2ab + b^2
   (a - b)^2 = a^2 - 2ab + b^2