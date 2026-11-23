# 8. Visualizations and Distribution Types

Source: `figures/v1` (`distribution_summary.parquet`, 10 features; figure PNGs). Scaling figures are in `scaling/v1` (`fig_08_*`). The scatter plots use a sample of 20,000 rows; the correlation heatmap uses Spearman.

## 8.1 Figure inventory

- Loaded-stage figures (`figures/v1`, 30 files, dpi 150): `fig_06_01` to `fig_06_30`. These cover minutes, n_steps, n_ingredients, calories, the %PDV nutrient columns, rating counts, top-20 ingredients and tags, long-tail plots for users and recipes, three scatter plots (ingredients vs minutes, calories vs rating, steps vs rating), a Spearman correlation heatmap (`fig_06_29`), and a box plot of rating by tag (`fig_06_30`).
- Scaling-stage figures (`scaling/v1`, 5 files): `fig_08_01` to `fig_08_05`, comparing raw and transformed calories and minutes.

## 8.2 Distribution type per feature

| Feature | Distribution type | Skewness | Kurtosis | p99 / median |
|---|---|---|---|---|
| minutes | long-tail | 481.29 | 231,636.80 | 22.60 |
| n_steps | long-tail | 2.03 | 10.69 | 3.33 |
| n_ingredients | right-skewed | 0.77 | 1.18 | 2.22 |
| calories | long-tail | 215.30 | 76,664.54 | 11.22 |
| total_fat_pdv | long-tail | 55.75 | 10,453.60 | 15.10 |
| sugar_pdv | long-tail | 402.83 | 182,232.62 | 45.67 |
| sodium_pdv | long-tail | 80.29 | 12,790.49 | 15.64 |
| protein_pdv | long-tail | 30.57 | 2,302.00 | 10.44 |
| saturated_fat_pdv | long-tail | 18.36 | 1,005.54 | 17.57 |
| carbohydrates_pdv | long-tail | 371.33 | 163,270.81 | 17.11 |

Nine of ten features are classed as long-tail. Only n_ingredients is classed as right-skewed (its skewness is below 1). The p99-to-median ratio ranges from 2.22 (n_ingredients) to 45.67 (sugar_pdv), so a few recipes sit far above the typical value.

These distribution types drive the transform choice in Section 10.

Sources: `figures/v1/distribution_summary.parquet`, `figures/v1/manifest.json`, `scaling/v1/manifest.json`.
