import os
import sys
import tempfile
import unittest
from pathlib import Path

# Configure writable cache directory for Matplotlib (critical for Codio / sandboxes)
if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# Robust Path & Dataset Discovery
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CWD = Path.cwd().resolve()
CODIO_WORKSPACE = Path("/home/codio/workspace")

SEARCH_DIRS = [
    SCRIPT_DIR,
    CWD,
    CODIO_WORKSPACE,
] + list(SCRIPT_DIR.parents) + list(CWD.parents)

for d in SEARCH_DIRS:
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))


def resolve_dataset(filename: str) -> Path:
    """
    Search for a dataset across all common Codio and local directory configurations:
    1. Same directory as this script
    2. Parent directories (handles tests running inside .guides/secure/ or subfolders)
    3. Current working directory and its parents
    4. Standard Codio workspace paths (/home/codio/workspace)
    5. Subdirectories (data/, data_scripts/)
    6. Recursive search fallback
    """
    candidates = []

    # 1. Script dir and parents
    candidates.append(SCRIPT_DIR / filename)
    candidates.append(SCRIPT_DIR / "data_scripts" / filename)
    candidates.append(SCRIPT_DIR / "data" / filename)
    for p in SCRIPT_DIR.parents:
        candidates.append(p / filename)
        candidates.append(p / "data_scripts" / filename)
        candidates.append(p / "data" / filename)

    # 2. CWD and parents
    candidates.append(CWD / filename)
    candidates.append(CWD / "data_scripts" / filename)
    candidates.append(CWD / "data" / filename)
    for p in CWD.parents:
        candidates.append(p / filename)
        candidates.append(p / "data_scripts" / filename)

    # 3. Codio workspace
    if CODIO_WORKSPACE.exists():
        candidates.append(CODIO_WORKSPACE / filename)
        candidates.append(CODIO_WORKSPACE / "data_scripts" / filename)
        candidates.append(CODIO_WORKSPACE / "data" / filename)

    # Direct match check
    for c in candidates:
        if c.is_file():
            return c.resolve()

    # 4. Recursive search fallback
    search_roots = [CWD, SCRIPT_DIR]
    if CODIO_WORKSPACE.exists():
        search_roots.insert(0, CODIO_WORKSPACE)

    for root in search_roots:
        if root.exists():
            try:
                for match in root.rglob(filename):
                    if match.is_file():
                        return match.resolve()
            except Exception:
                pass

    raise FileNotFoundError(
        f"Could not locate '{filename}'. Searched candidate locations:\n" +
        "\n".join(f"  - {c}" for c in candidates[:10])
    )


# ---------------------------------------------------------------------------
# Target Module Resolution (main.py vs solution.py)
# ---------------------------------------------------------------------------
target_module = None
if os.getenv("TEST_TARGET") == "solution":
    try:
        import solution as target_module
    except ImportError:
        pass

if target_module is None:
    try:
        import main as student_module
        if hasattr(student_module, "exhaustive_best_subset"):
            target_module = student_module
    except ImportError:
        pass

if target_module is None:
    try:
        import solution as target_module
    except ImportError:
        pass

if target_module is None:
    raise ImportError("Could not find either 'main.py' or 'solution.py' to test.")

exhaustive_best_subset = getattr(target_module, "exhaustive_best_subset", None)
forward_stepwise_selection = getattr(target_module, "forward_stepwise_selection", None)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------
class TestQ1ExhaustiveSubset(unittest.TestCase):
    """Unit tests for Question 1: Exhaustive Best Subset Selection (10 points total)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.coasters_path = resolve_dataset("coasters_numerical_only.csv")
            cls.df_coasters = pd.read_csv(cls.coasters_path)
            cls.X_coasters = cls.df_coasters.drop(columns=['speed_mph'])
            cls.y_coasters = cls.df_coasters['speed_mph']
            cls.setup_error = None
        except Exception as e:
            cls.coasters_path = None
            cls.df_coasters = None
            cls.X_coasters = None
            cls.y_coasters = None
            cls.setup_error = str(e)

    def _require_dataset(self):
        if self.df_coasters is None:
            self.fail(f"Could not load coasters_numerical_only.csv: {self.setup_error}")

    def test_q1_return_types(self):
        """Test that exhaustive_best_subset returns a tuple of feature names and a float BIC."""
        self._require_dataset()
        result = exhaustive_best_subset(self.X_coasters, self.y_coasters)
        self.assertIsNotNone(result, "exhaustive_best_subset returned None. Please implement the function.")
        self.assertIsInstance(result, (tuple, list), "Function must return a tuple (subset, bic).")
        self.assertEqual(len(result), 2, "Function must return exactly two elements: (subset, bic).")
        
        subset, bic = result
        self.assertIsInstance(subset, (tuple, list), "The best subset of features must be returned as a tuple (or list).")
        self.assertTrue(isinstance(bic, (float, np.floating)), f"The BIC must be a float, got {type(bic).__name__}.")
        self.assertGreater(len(subset), 0, "The best subset cannot be empty.")
        for col in subset:
            self.assertIn(col, self.X_coasters.columns, f"Returned feature '{col}' is not in X.columns.")

    def test_q1_synthetic_ground_truth(self):
        """Test that exhaustive_best_subset correctly isolates true signals from noise."""
        np.random.seed(42)
        n = 120
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        noise = np.random.randn(n)
        y = 3.0 * x1 - 2.5 * x2 + np.random.randn(n) * 0.4

        df_synth = pd.DataFrame({'x1': x1, 'x2': x2, 'noise': noise})
        y_synth = pd.Series(y, name='target')

        best_subset, best_bic = exhaustive_best_subset(df_synth, y_synth)
        self.assertIsNotNone(best_subset, "Function returned None on synthetic data.")
        self.assertEqual(
            set(best_subset),
            {'x1', 'x2'},
            f"Expected optimal subset to be ('x1', 'x2'), got {best_subset}."
        )

    def test_q1_coasters_optimal_subset(self):
        """Test that the optimal subset and BIC match the ground truth on coasters_numerical_only.csv."""
        self._require_dataset()
        best_subset, best_bic = exhaustive_best_subset(self.X_coasters, self.y_coasters)
        self.assertIsNotNone(best_subset, "Function returned None.")
        
        expected_features = {'height_ft', 'gforce_clean'}
        self.assertEqual(
            set(best_subset),
            expected_features,
            f"Optimal subset by BIC should be ('height_ft', 'gforce_clean'), but got {best_subset}."
        )
        self.assertAlmostEqual(
            best_bic,
            565.345,
            places=1,
            msg=f"Expected lowest BIC to be ~565.35, got {best_bic:.2f}."
        )

    def test_q1_input_immutability(self):
        """Test that exhaustive_best_subset does not mutate input X or y."""
        self._require_dataset()
        X_copy = self.X_coasters.copy()
        y_copy = self.y_coasters.copy()
        _ = exhaustive_best_subset(self.X_coasters, self.y_coasters)
        pd.testing.assert_frame_equal(self.X_coasters, X_copy, obj="Predictors DataFrame X was mutated.")
        pd.testing.assert_series_equal(self.y_coasters, y_copy, obj="Target Series y was mutated.")


class TestQ2ForwardStepwise(unittest.TestCase):
    """Unit tests for Question 2: Forward Stepwise Selection (15 points total)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.fifa_path = resolve_dataset("fifa_numerical_only.csv")
            cls.df_fifa = pd.read_csv(cls.fifa_path).head(500)
            cls.X_fifa = cls.df_fifa.drop(columns=['overall'])
            cls.y_fifa = cls.df_fifa['overall']
            cls.setup_error = None
        except Exception as e:
            cls.fifa_path = None
            cls.df_fifa = None
            cls.X_fifa = None
            cls.y_fifa = None
            cls.setup_error = str(e)

    def _require_dataset(self):
        if self.df_fifa is None:
            self.fail(f"Could not load fifa_numerical_only.csv: {self.setup_error}")

    def test_q2_return_types(self):
        """Test that forward_stepwise_selection returns a list of feature name strings."""
        self._require_dataset()
        selected = forward_stepwise_selection(self.X_fifa, self.y_fifa)
        self.assertIsNotNone(selected, "forward_stepwise_selection returned None. Please implement the function.")
        self.assertIsInstance(selected, list, "Function must return an ordered list of feature names.")
        self.assertGreater(len(selected), 0, "Selected features list cannot be empty.")
        for col in selected:
            self.assertIsInstance(col, str, f"Feature name '{col}' must be a string.")
            self.assertIn(col, self.X_fifa.columns, f"Feature '{col}' is not in X.columns.")

    def test_q2_synthetic_ground_truth(self):
        """Test forward selection on synthetic data with known hierarchical signal strengths."""
        np.random.seed(123)
        n = 200
        x_strong = np.random.randn(n)
        x_moderate = np.random.randn(n)
        x_noise = np.random.randn(n)
        y = 5.0 * x_strong + 2.0 * x_moderate + np.random.randn(n) * 0.5

        df_synth = pd.DataFrame({
            'x_noise': x_noise,
            'x_moderate': x_moderate,
            'x_strong': x_strong
        })
        y_synth = pd.Series(y, name='target')

        selected = forward_stepwise_selection(df_synth, y_synth)
        self.assertIsNotNone(selected, "Function returned None on synthetic data.")
        self.assertEqual(
            selected,
            ['x_strong', 'x_moderate'],
            f"Expected ['x_strong', 'x_moderate'], got {selected}."
        )

    def test_q2_fifa_greedy_selection_order(self):
        """Test that greedy steps on FIFA select the correct leading features."""
        self._require_dataset()
        selected = forward_stepwise_selection(self.X_fifa, self.y_fifa)
        self.assertIsNotNone(selected, "Function returned None.")
        self.assertGreater(len(selected), 1, "Must select at least 2 features.")
        
        # Step 1 should select movement_reactions (highest individual R^2_adj)
        self.assertEqual(
            selected[0],
            'movement_reactions',
            f"Step 1 should greedily select 'movement_reactions', got '{selected[0]}'."
        )
        
        # Step 2 should select attacking_short_passing
        self.assertEqual(
            selected[1],
            'attacking_short_passing',
            f"Step 2 should greedily select 'attacking_short_passing', got '{selected[1]}'."
        )

    def test_q2_fifa_halting_criterion(self):
        """Test that forward_stepwise_selection halts properly when Adj R^2 decreases."""
        self._require_dataset()
        selected = forward_stepwise_selection(self.X_fifa, self.y_fifa)
        self.assertIsNotNone(selected, "Function returned None.")
        
        # Must halt before selecting all features
        self.assertLess(
            len(selected),
            len(self.X_fifa.columns),
            "Algorithm failed to halt. It added all features instead of stopping when Adjusted R^2 dropped."
        )

        # Verify that no unselected feature can improve the final model's Adjusted R^2
        final_X = sm.add_constant(self.X_fifa[selected])
        final_adj_r2 = sm.OLS(self.y_fifa, final_X).fit().rsquared_adj

        remaining_features = [col for col in self.X_fifa.columns if col not in selected]
        for feature in remaining_features:
            test_X = sm.add_constant(self.X_fifa[selected + [feature]])
            test_adj_r2 = sm.OLS(self.y_fifa, test_X).fit().rsquared_adj
            self.assertLessEqual(
                test_adj_r2,
                final_adj_r2 + 1e-9,
                f"Algorithm halted prematurely: adding '{feature}' would improve Adjusted R^2 from {final_adj_r2:.4f} to {test_adj_r2:.4f}."
            )

    def test_q2_input_immutability(self):
        """Test that forward_stepwise_selection does not mutate input X or y."""
        self._require_dataset()
        X_copy = self.X_fifa.copy()
        y_copy = self.y_fifa.copy()
        _ = forward_stepwise_selection(self.X_fifa, self.y_fifa)
        pd.testing.assert_frame_equal(self.X_fifa, X_copy, obj="Predictors DataFrame X was mutated.")
        pd.testing.assert_series_equal(self.y_fifa, y_copy, obj="Target Series y was mutated.")


if __name__ == '__main__':
    unittest.main()
