Agas Tutorial
=============

What is Agas
^^^^^^^^^^^^

Agas is a small Python package allowing the user to find pairs of data
series (e.g., columns in a matrix, rows in a data frame) which optimally
fit flexible criteria for their aggregated values.

For example, Agas can be used to find the two rows which satisfy the
requirement of having similar median values, but vary in their standard
deviations.

Use cases
^^^^^^^^^

Agas can be useful to you when you want to explore your data, or present
how two units (e.g., patiants in your study) score similarly on some
aggregated measure and differently on another. You can select almost any
standard or custom aggregation function you wish, and set the weight of
the similarity function or divergence function flexibly.

Requirements
^^^^^^^^^^^^

NumPy and Pandas. That’s it.

Toy-data example
----------------

.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    import agas
    
    pd.set_option('display.precision', 2)
    pd.set_option("display.max_columns", 5)
    
    np.set_printoptions(precision=3)
    
    sns.set_context('notebook')
    
    
    def normalize_data(s):
        """Normalize an array of values between 0 and 1.
        """
        return (s - s.min()) / (s.max() - s.min())

Our data is a matrix of size 4 X 4. Given 4 rows, there are 6 unique
pairs of rows (``N(N - 1) / 2``).

.. code:: ipython3

    a = np.vstack([
        [0, -1, -1, -1],
        [1, 2, 3, 4],
        [6, 9, 9, 9],
        [8, 9, 10, 11]])

We want to find pair of rows which have the most similar median values
at the same time as having the most different standard deviations.

.. code:: ipython3

    maximize_similarity = np.median
    maximize_divergence = np.std

This matrix shows the median values on each row, and their normalized
values (right column):

.. code:: ipython3

    a_medians = maximize_similarity(a, axis=1)
    np.stack([a_medians, normalize_data(a_medians)], axis=1)




.. parsed-literal::

    array([[-1.   ,  0.   ],
           [ 2.5  ,  0.333],
           [ 9.   ,  0.952],
           [ 9.5  ,  1.   ]])



This matrix shows the standard deviations for each row, and their
normalized values (right column):

.. code:: ipython3

    a_sds = maximize_divergence(a, axis=1)
    np.stack([a_sds, normalize_data(a_sds)], axis=1)




.. parsed-literal::

    array([[0.433, 0.   ],
           [1.118, 0.791],
           [1.299, 1.   ],
           [1.118, 0.791]])



Here it is possible to guess that the pair of rows from ``a`` with the
highest similarity in medians and highest divergence in SDs is the pair
of two first rows, as they have relatively close median values (0 and
0.33, when normalized) and quite different standard deviations (0 and
0.79, normalized).

As you’ll see below, this is also the result obtained using ``Agas``.

Calling ``agas.pair_from_array`` below returns a tuple of two arrays: -
A 2-D array of size 6 X 2, representing the pairs of row indices. - A
1-D array of the optimality scores of each pair (0 is the best, 1 is the
worst).

.. code:: ipython3

    indices, scores = agas.pair_from_array(a, maximize_similarity,
                                           maximize_divergence, return_filter='all')
    print(f"The optimal score: {scores[0]} - rows {indices[0]}")
    print(f"The least optimal score: {scores[-1]} - rows {indices[-1]}")


.. parsed-literal::

    The optimal score: 0.0 - rows [0 1]
    The least optimal score: 1.0 - rows [1 3]
    

Agas can return the matrix of optimality scores, using the
``return_matrix`` argument:

.. code:: ipython3

    optimality_scores_matrix = agas.pair_from_array(
        a, maximize_similarity, maximize_divergence, return_matrix=True)

Viewed as a matrix, the scores match the conclusions from above where
the pairing of rows 0 and 1 provides an optimal fit, while this of rows
1 and 2 provides the least favorable pair.

The diagonal of the matrix is un-colored, as ``Agas`` removes pairings
of each row with itself:

.. code:: ipython3

    g = sns.heatmap(optimality_scores_matrix, square=True, annot=True,
                    linewidths=1, linecolor='black', cmap='viridis', fmt=".2f")
    g.set_facecolor('grey')



.. image:: tutorial_files%5Ctutorial_17_0.png


``Agas`` can weight differently the similarity and divergence functions
using the ``similarity_weight`` argument. Here we favor similaritry in
median value over divergence in standard deviation.

.. code:: ipython3

    medians_biased_scores_mat = agas.pair_from_array(
        a, maximize_similarity, maximize_divergence, similarity_weight=0.75,
        return_matrix=True)
    g = sns.heatmap(medians_biased_scores_mat, square=True, annot=True,
                    linewidths=1, linecolor='black', cmap='viridis', fmt=".2f")
    g.set_facecolor('grey')



.. image:: tutorial_files%5Ctutorial_19_0.png


Real-world example
------------------

We load a dataset containing the GDP values for different countries and
regions, 1968-2016. While a relatively a small dataset, ``agas`` is
useful here as it is impossible to manually compare all pairs of rows.

.. code:: ipython3

    url = 'https://datahub.io/core/gdp/r/gdp.csv'
    df = pd.read_csv(url)
    df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Country Name</th>
          <th>Country Code</th>
          <th>Year</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Arab World</td>
          <td>ARB</td>
          <td>1968</td>
          <td>2.58e+10</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Arab World</td>
          <td>ARB</td>
          <td>1969</td>
          <td>2.84e+10</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Arab World</td>
          <td>ARB</td>
          <td>1970</td>
          <td>3.14e+10</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Arab World</td>
          <td>ARB</td>
          <td>1971</td>
          <td>3.64e+10</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Arab World</td>
          <td>ARB</td>
          <td>1972</td>
          <td>4.33e+10</td>
        </tr>
      </tbody>
    </table>
    </div>



Change the unit of Value (GDP) from $ to trillions, for easier
readability

.. code:: ipython3

    df['Value'] /= 1e12

Remove the top and bottom 2.5 percentiles

.. code:: ipython3

    total_per_country = df.groupby('Country Name')['Value'].sum()
    non_outliers = total_per_country[total_per_country.between(
        *total_per_country.quantile([0.025, 0.975]))].index
    df = df.loc[df['Country Name'].isin(non_outliers)]

Considerable number of countries and regions have no data until the
1990s:

.. code:: ipython3

    ax = plt.scatter(df['Year'].sort_values().unique(),
                     df.groupby('Year')['Country Name'].nunique() / df[
                         'Country Name'].nunique())
    plt.gca().set(xlabel='Year', ylabel='Proportion of non-missing data')




.. parsed-literal::

    [Text(0.5, 0, 'Year'), Text(0, 0.5, 'Proportion of non-missing data')]




.. image:: tutorial_files%5Ctutorial_28_1.png


Pivoting the data frame, as ``Agas`` 0.0.1 only handles wide-format
data.

.. code:: ipython3

    wide_df = df.loc[df['Year'].gt(1990)].pivot(columns='Year', values='Value',
                                                index='Country Name')
    wide_df.sample(5, random_state=42)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>Year</th>
          <th>1991</th>
          <th>1992</th>
          <th>...</th>
          <th>2015</th>
          <th>2016</th>
        </tr>
        <tr>
          <th>Country Name</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Bolivia</th>
          <td>5.34e-03</td>
          <td>5.64e-03</td>
          <td>...</td>
          <td>3.30e-02</td>
          <td>3.38e-02</td>
        </tr>
        <tr>
          <th>Antigua and Barbuda</th>
          <td>4.82e-04</td>
          <td>4.99e-04</td>
          <td>...</td>
          <td>1.36e-03</td>
          <td>1.46e-03</td>
        </tr>
        <tr>
          <th>Monaco</th>
          <td>2.48e-03</td>
          <td>2.74e-03</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>Sub-Saharan Africa (excluding high income)</th>
          <td>3.19e-01</td>
          <td>3.13e-01</td>
          <td>...</td>
          <td>1.60e+00</td>
          <td>1.51e+00</td>
        </tr>
        <tr>
          <th>Virgin Islands (U.S.)</th>
          <td>1.67e-03</td>
          <td>1.77e-03</td>
          <td>...</td>
          <td>3.77e-03</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    <p>5 rows × 26 columns</p>
    </div>



Here we use ``agas.pair_from_wide_df``, which behaves similary to
``agas.pair_from_array`` used above.

.. code:: ipython3

    indices, scores = agas.pair_from_wide_df(
        wide_df, np.nanstd, np.median, similarity_weight=0.7, return_filter='all')


.. parsed-literal::

    c:\users\eitan hemed\onedrive - university of haifa\phd\python_projects\agas\agas\_from_numpy.py:266: RuntimeWarning: The result of aggregating the input scores using the function median resulted in 45 NaN scores.
      warnings.warn(f"The result of aggregating the input scores using the "
    

Select the optimal pair - two entries which are most similar in their
variances (``np.nanstd``) and most divergent in their median GDP values
(``np.nanmedian``).

.. code:: ipython3

    wide_df.iloc[indices[0].flatten()].agg([np.nanmedian, np.nanstd], axis=1)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>nanmedian</th>
          <th>nanstd</th>
        </tr>
        <tr>
          <th>Country Name</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>China</th>
          <td>1.81</td>
          <td>3.72</td>
        </tr>
        <tr>
          <th>United States</th>
          <td>11.89</td>
          <td>3.86</td>
        </tr>
      </tbody>
    </table>
    </div>



China and United States show similar variability in their GDP, but
different median values. It is interesting to explore other
relationships between pairs of data points, as we do using the plot
below.

.. code:: ipython3

    # Normalized aggregated data
    sds = normalize_data(wide_df.std(ddof=0, axis=1))
    medians = normalize_data(wide_df.mean(axis=1))
    
    # Indices of interesting data points, the most otpimal, least optimal and
    country_indices = np.arange(0, wide_df.index.nunique())
    scores_of_interest = [0, scores[scores.size // 2], 1]
    scores_of_interest_indices = indices[np.in1d(scores, scores_of_interest)]
    country_indices = country_indices[
        ~np.in1d(country_indices, scores_of_interest_indices)]
    
    # Plot asthetics
    plot_colors = ['deepskyblue', 'purple', 'green', 'orange']
    scores_of_interest_labels = ['Optimal', 'Middling', 'Least Optimal']
    scores_of_interest_line_styles = ["solid", "dotted", "dashed"]
    
    colors = np.repeat(plot_colors[0], wide_df.shape[0])
    alphas = np.ones_like(colors).astype(float) * 0.1

The plot below shows the data, first raw and then noramlized and
aggregated. The most optimal pair was United States and china, which
have similar variance and divergent median GDP as shown above.

the least optimal pair was Israel and Swaziland, which show divergent
standard deviations and similar medians.

.. code:: ipython3

    fig, axs = plt.subplots(2, 2, figsize=(10, 8),
                            gridspec_kw={'height_ratios': [8, 1]})
    raw_data_ax, norm_aggrgated_data_ax, _ax_to_remoev, legend_ax = axs.flat
    
    # Plot the data excluding the data matching the points of interest
    raw_data_ax.plot(wide_df.iloc[country_indices].T, c='deepskyblue', alpha=0.5,
                     lw=0.5)
    raw_data_ax.set(xlabel='Year', ylabel='GDP (Trillion $), Log-scale')
    
    for current_row_indices, _color, lab, ls in zip(scores_of_interest_indices,
                                                    plot_colors[1:],
                                                    scores_of_interest_labels,
                                                    scores_of_interest_line_styles
                                                    ):
        country_labels = wide_df.index[current_row_indices]
        colors[current_row_indices] = _color
        alphas[current_row_indices] = 1
    
        raw_data_ax.plot(wide_df.iloc[current_row_indices].T,
                         c=_color, label=f'{lab} \n' + ' | '.join(country_labels),
                         lw=3, ls=ls)
    
        for idx, l in zip(current_row_indices, country_labels):
            norm_aggrgated_data_ax.annotate(xy=[medians[idx], sds[idx] * 1.25],
                                            text=l,
                                            color=_color, ha='center')
    
    # Scatter the normalized and aggregated data
    norm_aggrgated_data_ax.scatter(medians, sds, c=colors, alpha=alphas)
    norm_aggrgated_data_ax.set(xlabel='Normalized Median',
                               ylabel='Normalized SD, Log-scale')
    
    # # Remove duplicate entries in the legend and place in the lower row of axis
    handles, country_labels = raw_data_ax.get_legend_handles_labels()
    by_label = dict(zip(country_labels, handles))
    legend_ax.legend(by_label.values(), by_label.keys())
    legend_ax.axis('off')
    _ax_to_remoev.remove()
    
    norm_aggrgated_data_ax.set_yscale('log')
    raw_data_ax.set_yscale('log')
    
    fig.tight_layout()



.. image:: tutorial_files%5Ctutorial_38_0.png

