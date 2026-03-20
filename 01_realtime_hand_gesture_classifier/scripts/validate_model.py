"""
scripts/validate_model.py
─────────────────────────
Called by the model_validation GitHub Actions workflow.
Loads the trained MLP model and runs automated checks.

Exit codes:
  0 = all checks passed
  1 = one or more checks failed (CI step will be marked as failed)

Usage:
  python scripts/validate_model.py
  python scripts/validate_model.py --min-accuracy 0.80
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import joblib

# Add project root to path so we can import from components/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from components.augmentation import augment_batch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-accuracy", type=float, default=0.70,
                        help="Minimum accuracy threshold (default: 0.70)")
    return parser.parse_args()


def load_model(model_path: Path):
    """Load model from disk; return None with an error if not found."""
    if not model_path.exists():
        print(f"[SKIP] Model not found at {model_path} — skipping validation.")
        print("       Train a model first via the Streamlit app.")
        sys.exit(0)  # not a failure — just nothing to validate yet

    print(f"[OK]   Loading model from {model_path}")
    return joblib.load(model_path)


def check_deserialisation(model) -> bool:
    """Check 1: model loads and has the expected sklearn Pipeline structure."""
    passed = (
        hasattr(model, "predict") and
        hasattr(model, "predict_proba") and
        hasattr(model, "named_steps") and
        "mlp" in model.named_steps and
        "scaler" in model.named_steps
    )
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Deserialisation check — Pipeline with scaler + mlp")
    return passed


def check_output_classes(model) -> bool:
    """Check 2: model has been trained on at least 2 classes."""
    classes = model.classes_
    passed = len(classes) >= 2
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Output classes: {sorted(classes.tolist())} ({len(classes)} classes)")
    return passed


def check_input_shape(model) -> bool:
    """Check 3: model accepts 63-feature input without error."""
    try:
        dummy = np.random.randn(1, 63).astype(np.float32)
        _ = model.predict_proba(dummy)
        print("[PASS] Input shape check — accepts (1, 63) feature vector")
        return True
    except Exception as e:
        print(f"[FAIL] Input shape check — {e}")
        return False


def check_prediction_diversity(model) -> bool:
    """
    Check 4: model doesn't always predict the same class on random inputs.
    A degenerate/collapsed model would fail this.
    """
    n_samples = 200
    dummy = np.random.randn(n_samples, 63).astype(np.float32)
    preds = model.predict(dummy)
    unique = np.unique(preds)
    passed = len(unique) >= min(2, len(model.classes_))
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} Prediction diversity — {len(unique)} distinct classes predicted on {n_samples} random inputs")
    return passed


def check_accuracy_on_synthetic(model, min_accuracy: float) -> bool:
    """
    Check 5: accuracy on a synthetic test set derived from the training data.

    Since we don't have a fixed held-out set in CI, we generate a small
    synthetic dataset by augmenting known-good landmark patterns and
    checking that the model correctly classifies them.

    This is a sanity check, not a rigorous evaluation — that happens during
    training in the Streamlit app.
    """
    from components.dataset import load_dataset

    X, y = load_dataset()
    if X is None or len(X) == 0:
        print("[SKIP] No landmark data found — skipping accuracy check.")
        return True

    # Augment to create a small test set
    X_test, y_test = augment_batch(X, y, n_copies=3)

    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    passed = accuracy >= min_accuracy
    status = "[PASS]" if passed else "[FAIL]"
    print(
        f"{status} Accuracy on augmented test set: {accuracy:.1%} "
        f"(threshold: {min_accuracy:.1%}, n={len(y_test)})"
    )
    return passed


def main():
    args = parse_args()

    model_path = ROOT / "models" / "mlp_pipeline.pkl"
    model = load_model(model_path)

    print("\nRunning validation checks...")
    print("─" * 50)

    results = [
        check_deserialisation(model),
        check_output_classes(model),
        check_input_shape(model),
        check_prediction_diversity(model),
        check_accuracy_on_synthetic(model, args.min_accuracy),
    ]

    print("─" * 50)
    passed = sum(results)
    total  = len(results)
    print(f"\nResult: {passed}/{total} checks passed")

    if all(results):
        print("✅ Model validation PASSED")
        sys.exit(0)
    else:
        print("❌ Model validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
