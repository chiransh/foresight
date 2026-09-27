# Do the models fail on the same stores?

All three models scored over the same 3 expanding-window folds and the same 1,115 stores, compared store by store rather than in aggregate. The question is whether one model's hard stores are also the others', because that decides whether there is anything to gain from combining them.

## Median per-store WAPE

| Model | Median store WAPE |
|---|---|
| lightgbm | 8.55 |
| prophet | 11.05 |
| nhits | 11.91 |

## Agreement between models

Worst decile means each model's own 111 highest-WAPE stores, so the overlap between two of them is a count of shops both models put in their worst tenth.

lightgbm and nhits show moderate rank agreement across stores (Spearman +0.65). They also agree on which shops are hardest: 59 of each model's worst 111 stores are the same stores, 53 percent where chance alone would give 10.

lightgbm and prophet show moderate rank agreement across stores (Spearman +0.52). Their worst deciles overlap on 42 of 111 stores, 38 percent against the 10 chance would give. More than chance, and well short of the same set of shops.

nhits and prophet show moderate rank agreement across stores (Spearman +0.68). Their worst deciles overlap on 49 of 111 stores, 44 percent against the 10 chance would give. More than chance, and well short of the same set of shops.

## Which model wins each store

| Model | Stores won | Share |
|---|---|---|
| lightgbm | 1,039 | 93.2% |
| nhits | 4 | 0.4% |
| prophet | 72 | 6.5% |

## What a per-store choice would be worth

The best single model over all folds is lightgbm, at 8.66 WAPE across 1,115 stores. Giving every store the model that turned out to suit it would reach 8.62. That second figure is not a result: the model is picked using the same window it is then scored on, so it is a bound on what the honest version below could find.

| Decision | Chose on folds | Deployed single model | Its WAPE | Selection WAPE | Gain | 95% interval |
|---|---|---|---|---|---|---|
| fold 2 | 1 | lightgbm | 8.43 | 8.43 | +0.0% | -0.3% to +0.3% |
| fold 3 | 1, 2 | lightgbm | 8.13 | 8.21 | -1.0% | -2.1% to -0.3% |

There is no per-store choice to make here. A perfect hindsight choice would cut WAPE by 0.04 points, because lightgbm is the better model on 93 percent of stores, so even the bound is nearly zero. Trying it anyway costs accuracy: the selection is reliably worse in 1 of 2 decisions. The reason is in the switched stores: of those moved off the champion, 38 to 50 percent actually improved, so the past window is close to a coin flip about which model suits a store.

## Caveats

- All three models run at default hyperparameters, so this compares the stores each default configuration finds hard, not the stores each model family is intrinsically bad at.
- A per-store choice is the cheapest way to combine models and not the only one. Averaging or stacking the forecasts could help where selection does not, and needs the row-level predictions this rig discards.
- Stores any model declined to score are dropped from every model, so the comparison is over a common set rather than each model's best case.
- The models agree on how much of each store they scored to within 0.00 percent, so the per-store figures are comparable rather than three different windows.
