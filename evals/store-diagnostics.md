# Where the forecast fails

Per-store error for the served model family across 3 expanding-window folds, over 1,115 stores. Overall WAPE is 8.66, which is a single number covering a wide spread of stores. This is that spread.

## The spread

| | WAPE |
|---|---|
| Best store | 5.56 |
| 25th percentile | 7.82 |
| Median | 8.55 |
| 75th percentile | 9.53 |
| 90th percentile | 10.72 |
| Worst store | 21.95 |

The worst 111 stores, a tenth of the estate, carry 12 percent of the total absolute error while making 9 percent of the sales, a ratio of 1.40. That is mild concentration: the worst tenth are worse than their share of sales, but not by enough to make them a separate problem from the rest.

The distribution is tight rather than long-tailed: the 90th percentile store is only 1.25 times the median, and 23 of 1,115 stores sit above 1.5 times the median. The worst store is 3.9 times the best, so the outliers are real but there are very few of them.

## By store type

| Store type | Stores | Median WAPE | Share of total error |
|---|---|---|---|
| a | 602 | 8.41 | 51.8% |
| b | 17 | 8.25 | 2.7% |
| c | 148 | 8.92 | 13.6% |
| d | 348 | 8.80 | 31.9% |

## By sales volume

Quintiles of mean daily sales, lowest first.

| Quintile | Stores | Median daily sales | Median WAPE |
|---|---|---|---|
| 1 | 223 | 4,526 | 9.41 |
| 2 | 223 | 5,538 | 8.88 |
| 3 | 223 | 6,590 | 8.46 |
| 4 | 223 | 7,621 | 8.36 |
| 5 | 223 | 9,686 | 7.68 |

## What predicts a badly forecast store

Sales volume shows a clear rank correlation with error (-0.43): more of it means lower error.

Closure rate shows no rank correlation with error (+0.17).

Length of trading history shows no rank correlation with error (-0.20).

## The ten worst stores

| Store | WAPE | Mean daily sales | Type | Assortment | Closed days |
|---|---|---|---|---|---|
| 183 | 21.95 | 4,821 | a | a | 19% |
| 299 | 17.40 | 6,235 | d | c | 6% |
| 292 | 16.93 | 5,774 | a | a | 19% |
| 909 | 16.74 | 10,729 | a | c | 20% |
| 971 | 16.17 | 8,270 | c | a | 18% |
| 675 | 15.91 | 4,278 | a | a | 17% |
| 876 | 15.22 | 9,229 | a | a | 19% |
| 931 | 15.22 | 3,589 | a | c | 7% |
| 39 | 14.87 | 4,853 | a | a | 17% |
| 530 | 14.53 | 4,634 | a | c | 4% |

The closure column is worth reading carefully: the ten worst stores close 18 percent of days against an estate median of 17 percent. They are ordinary in that respect, so closures are not what makes them hard to forecast.

## Caveat

These are the default hyperparameters and the same folds as the model comparison, so the numbers line up with it. A per-store breakdown says where error sits; it does not say the model could be made better there, and some of these shops may simply be less predictable than others.
