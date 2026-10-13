import os
import tempfile
from itertools import combinations
from pathlib import Path

# Ensure writable matplotlib cache directory (prevents Codio sandbox permission warnings)
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd
import statsmodels.api as sm


def exhaustive_best_subset(X, y):
    """
    Evaluate all 2^p - 1 non-empty predictor combinations using statsmodels OLS with intercept.
    
    Parameters:
      X (pd.DataFrame): Predictor variables (features).
      y (pd.Series): Target response variable.
      
    Returns:
      tuple: (optimal_subset_tuple, lowest_bic_float)
        - optimal_subset_tuple (tuple of str): Column names of the subset with the lowest BIC.
        - lowest_bic_float (float): The BIC value corresponding to the best subset model.
    """
    # YOUR CODE HERE
    pass


def forward_stepwise_selection(X, y):
    """
    Implement a greedy forward stepwise selection algorithm using statsmodels OLS with intercept.
    Iteratively add the single predictor that most improves the model's Adjusted R^2.
    Halt execution exactly when adding any remaining predictor decreases Adjusted R^2.
    
    Parameters:
      X (pd.DataFrame): Predictor variables (features).
      y (pd.Series): Target response variable.
      
    Returns:
      list of str: The ordered list of selected feature names.
    """
    # YOUR CODE HERE
    pass


if __name__ == '__main__':
    base_dir = Path(__file__).resolve().parent

    # Q1 Execution
    print("--- Q1: Exhaustive Best Subset Selection ---")
    coasters_csv = base_dir / "coasters_numerical_only.csv"
    if not coasters_csv.exists():
        coasters_csv = Path("coasters_numerical_only.csv")

    try:
        df_coasters = pd.read_csv(coasters_csv)
        y_c = df_coasters['speed_mph']
        X_c = df_coasters.drop(columns=['speed_mph'])
        
        result = exhaustive_best_subset(X_c, y_c)
        if result is not None:
            best_subset, best_bic = result
            print(f"Optimal Coaster Predictors (Lowest BIC): {best_subset}")
            print(f"Lowest BIC Value: {best_bic:.2f}\n")
        else:
            print("exhaustive_best_subset returned None. Complete the function above.\n")
    except FileNotFoundError:
        print(f"Dataset 'coasters_numerical_only.csv' not found.\n")

    # Q2 Execution
    print("--- Q2: Forward Stepwise Selection ---")
    fifa_csv = base_dir / "fifa_numerical_only.csv"
    if not fifa_csv.exists():
        fifa_csv = Path("fifa_numerical_only.csv")

    try:
        df_fifa = pd.read_csv(fifa_csv)
        y_f = df_fifa['overall']
        X_f = df_fifa.drop(columns=['overall'])
        
        print("Running forward stepwise selection on FIFA data...")
        selected = forward_stepwise_selection(X_f, y_f)
        if selected is not None:
            print(f"Selected Features ({len(selected)} total, ordered):")
            print(selected)
        else:
            print("forward_stepwise_selection returned None. Complete the function above.\n")
    except FileNotFoundError:
        print(f"Dataset 'fifa_numerical_only.csv' not found.\n")
