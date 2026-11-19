---
title: "Q1: Exhaustive Best Subset Selection (O(2^p))"
type: coding_question
points: 10
dataset: coasters_numerical_only.csv
---

# ICE 5: Model Selection — Best Subset & Forward Stepwise

In this assignment, you will implement two classical model selection strategies in `main.py` using `pandas` and `statsmodels.api`:
1. **Question 1 (10 pts):** Exhaustive Best Subset Selection evaluated using the Bayesian Information Criterion (BIC).
2. **Question 2 (15 pts):** Greedy Forward Stepwise Selection evaluated using Adjusted $R^2$.

All files are located flat in the root directory.

---

## Datasets Overview

Both datasets are provided directly in your working directory:

1. **`coasters_numerical_only.csv`** (72 observations, 4 numeric columns):
   - **Target ($y$):** `speed_mph`
   - **Predictors ($X$):** `height_ft`, `inversions_clean`, `gforce_clean`
   - With $p = 3$ predictors, there are $2^3 - 1 = 7$ non-empty candidate models, making exhaustive search trivial.

2. **`fifa_numerical_only.csv`** (17,107 observations, 33 numeric columns):
   - **Target ($y$):** `overall` (player overall rating)
   - **Predictors ($X$):** 32 granular sub-stats (e.g. `movement_reactions`, `skill_ball_control`, `attacking_short_passing`, etc.)
   - With $p = 32$ predictors, exhaustive search would require evaluating $2^{32} - 1 \approx 4.29 \times 10^9$ models. Forward stepwise selection reduces this to at most $p(p+1)/2 = 528$ models.

---

## Question 1: Exhaustive Best Subset Selection (10 points)

### Task
Implement a pure function `exhaustive_best_subset(X, y)` in `main.py`:

```python
def exhaustive_best_subset(X, y):
    """
    Evaluate all 2^p - 1 non-empty subsets of predictors using statsmodels OLS with intercept.
    
    Parameters:
      X (pd.DataFrame): Predictor variables (features).
      y (pd.Series): Target response variable.
      
    Returns:
      tuple: (optimal_subset_tuple, lowest_bic_float)
    """
```

### Specific Requirements & Implementation Details
1. **Model Specification:**
   - For every subset of features of size $k \in \{1, 2, \dots, p\}$, extract those columns from `X`.
   - **Always add an intercept** using `statsmodels.api.add_constant(...)`:
     ```python
     X_subset = sm.add_constant(X[list(subset)])
     model = sm.OLS(y, X_subset).fit()
     ```
2. **Evaluation Metric:**
   - Extract the model's Bayesian Information Criterion (BIC) via `model.bic`.
   - Find the subset of predictors that minimizes the BIC.
3. **Pure Function (Immutability):**
   - Do not mutate `X` or `y` in-place.
4. **Return Format:**
   - Return a 2-element `tuple`: `(optimal_subset, lowest_bic)`.
   - `optimal_subset` must be a `tuple` of column names (e.g., `('height_ft', 'gforce_clean')`).
   - `lowest_bic` must be a `float` (e.g., `565.35`).

### Driver Code (`if __name__ == '__main__':`)
Load `coasters_numerical_only.csv`, set `speed_mph` as $y$ and the remaining 3 columns as $X$. Call `exhaustive_best_subset(X, y)` and print the best feature tuple and its BIC.

---

## Question 2: Forward Stepwise Selection (15 points)

---
title: "Q2: Forward Stepwise Selection (Intractable Feature Space)"
type: coding_question
points: 15
dataset: fifa_numerical_only.csv
---

### Task
Implement a pure function `forward_stepwise_selection(X, y)` in `main.py`:

```python
def forward_stepwise_selection(X, y):
    """
    Greedy forward stepwise selection using statsmodels OLS with intercept.
    Iteratively adds the single predictor that most improves Adjusted R^2.
    Halts execution when adding any remaining predictor decreases Adjusted R^2.
    
    Parameters:
      X (pd.DataFrame): Predictor variables (features).
      y (pd.Series): Target response variable.
      
    Returns:
      list: Ordered list of selected feature names.
    """
```

### Specific Requirements & Implementation Details
1. **Algorithm Flow:**
   - Initialize `selected_features = []` and a candidate pool containing all column names from `X`.
   - At each greedy step, consider each remaining candidate feature in the pool added to `selected_features`.
   - Fit an OLS regression model with intercept using `sm.add_constant`:
     ```python
     X_subset = sm.add_constant(X[test_features])
     model = sm.OLS(y, X_subset).fit()
     ```
   - Record the model's Adjusted $R^2$ using `model.rsquared_adj`.
   - Determine which candidate feature produces the highest Adjusted $R^2$ for this step.
2. **Halting Condition:**
   - If adding the best candidate feature strictly increases the current model's Adjusted $R^2$, append it to `selected_features`, remove it from the pool, and proceed to the next step.
   - If adding any remaining candidate feature results in a lower or equal Adjusted $R^2$, halt immediately and stop searching.
3. **Pure Function (Immutability):**
   - Do not mutate `X` or `y` in-place.
4. **Return Format:**
   - Return an ordered `list` of strings containing the selected feature names in the exact sequence they were added (e.g., `['movement_reactions', 'skill_ball_control', ...]`).

### Driver Code (`if __name__ == '__main__':`)
Load `fifa_numerical_only.csv`, set `overall` as $y$ and the remaining 32 columns as $X$. Call `forward_stepwise_selection(X, y)` and print the ordered list of selected features.


