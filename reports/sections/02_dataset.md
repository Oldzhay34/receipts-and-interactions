# Dataset Description

## Source files

Only the two raw files of the Food.com dataset are used: `RAW_recipes.csv` (the recipe catalogue)
and `RAW_interactions.csv` (user reviews with a star rating). The dataset also ships
pre-processed `PP_` files and ready-made train / validation / test files; they are not used. They
were produced by a third party on a filtered subset of the users, with tokenisation and splits we
cannot reproduce or stratify, while this project needs both its own cleaning decisions and its own
leakage-free splits (Section {{sec:sampling}}).

## Loading

**Finding.** In the CSV files, list-valued columns (`tags`, `steps`, `ingredients`) are stored as
text and the seven nutrition values are packed into one list-valued text column. **Decision.** The
loader parses the list columns into Python lists, splits `nutrition` into seven numeric columns
(`calories` and six "% daily value" columns), parses dates and down-casts integers; only empty
fields count as missing, so literal text such as "None" or "NA" inside a recipe name or review is
kept. Both tables are validated against a schema before they are written as the `loaded` stage.
**Rationale.** Every later stage works on typed columns, and a schema violation stops the
pipeline instead of propagating silently.

After loading, the recipe table has 18 columns and the interaction table 5
<!-- src: outputs/exploration/v1/column_types.parquet --> ({{tab:column_types}}):

| Table | Columns | Type |
|---|---|---|
| recipes | `id`, `contributor_id` | identifiers |
| recipes | `name`, `description` | free text |
| recipes | `submitted` | date |
| recipes | `minutes`, `n_steps`, `n_ingredients` | numeric (time, counts) |
| recipes | `calories`, `total_fat_pdv`, `sugar_pdv`, `sodium_pdv`, `protein_pdv`, `saturated_fat_pdv`, `carbohydrates_pdv` | numeric (nutrition; `_pdv` = % daily value) |
| recipes | `tags`, `steps`, `ingredients` | lists of strings |
| interactions | `user_id`, `recipe_id` | identifiers |
| interactions | `date` | date |
| interactions | `rating` | ordinal 0–5; 0 = no stars given |
| interactions | `review` | free text |

<!-- src: outputs/exploration/v1/column_types.parquet (table, column, type_class, note) -->

## Size and coverage

| Quantity | Value |
|---|---|
| Recipes | 231,637 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Recipe contributors | 27,926 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Recipes submitted | 1999-08-06 – 2018-12-04 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Interactions (reviews) | 1,132,367 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Users | 226,570 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Interaction dates | 2000-01-25 – 2018-12-20 <!-- src: outputs/exploration/v1/basic_counts.parquet --> |
| Distinct ingredient strings | 14,942 <!-- src: outputs/quality/v2/dqr_lists.parquet --> |
| Distinct tags | 552 <!-- src: outputs/quality/v2/dqr_lists.parquet --> |

## Referential integrity

**Finding.** Every interaction refers to an existing recipe (0 orphan interactions), there are no
exact duplicate interactions and no user reviewed the same recipe twice (0 repeated user–recipe
pairs), and every recipe has at least one interaction (0 recipes without interactions)
<!-- src: outputs/exploration/v1/basic_counts.parquet -->. **Decision.** No row is dropped at
load time. **Rationale.** The two tables are consistent as delivered; the remaining problems are
value-level (Sections {{sec:missing}} and {{sec:outliers}}), not structural.

## Per-recipe lists

Ingredient lists contain 1 to 43 items with a median of 9; tag lists contain 1 to 73 tags with a
median of 17 <!-- src: outputs/quality/v2/dqr_lists.parquet --> ({{tab:dqr_lists}}). Ingredient
strings are free text written by the contributors, so the 14,942 distinct strings
<!-- src: outputs/quality/v2/dqr_lists.parquet --> include many variants of the same ingredient
("eggs" / "egg", "garlic cloves" / "garlic") (Section {{sec:exploration}}); they are normalised
before encoding (Section {{sec:ingredients}}).

## Versioning

Every stage writes a new numbered version (`outputs/<stage>/vN/`) with a manifest that records
row counts, schema, SHA-256 checksum of every file, the code commit and the input versions; an
existing version is never overwritten. The numbers in this report cite the exact file they were
read from.
