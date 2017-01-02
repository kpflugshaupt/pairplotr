![Alt text](./pairplotr_demo.png?raw=true)
# How
## Installation
For now, simply clone the respository and link to the location in your code. 

## Use
See the [demo](pairplotr_demo.ipynb) for use of pairplotr.

# What
Pairplotr is a Python library used to graph combinations of numerical and categorical data in a pair plot,
similar to Seaborn's pairplot(), given a cleaned Pandas dataframe with a mixture of categorical and numerical
values.

Here are the formats for Row feature|Column feature combinations in either on- or off-diagonal cells: 

- On-diagonal:        
  - Categorical|Categorical:
    - Value counts of feature values ordered by ascending value count and colored by feature values
  - Numerical|Numerical:
    - Histogram of feature w/ no coloring (or by desired label)
- Off-diagonal:
  - Categorical|Categorical:
    - Stacked value count of row feature values colored by column feature values
  - Categorical|Numerical:
    - Histograms of column feature for each row feature value colored by row feature value
  - Numerical|Numerical:
    - Scatter plot of row feature values vs column feature values w/ no coloring (or by desired label)

# Why
The available tools I've found don't seem to be able to combine numerical and categorical feature data
in a quick and easy way and I wanted to customize the comparisons as the plot types I find most useful.