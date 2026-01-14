import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def sample_data():
    """
    Load a small deterministic sample of the census dataset for testing.
    Keeps tests fast and repeatable.
    """
    project_root = Path(__file__).resolve().parent
    csv_path = project_root / "data" / "census.csv"

    df = pd.read_csv(csv_path)

    # small sample for speed, deterministic
    df = df.sample(n=1000, random_state=42).reset_index(drop=True)
    return df


@pytest.fixture(scope="module")
def processed_train_test(sample_data):
    """
    Split + process data once and reuse across tests.
    """
    train_df, test_df = train_test_split(
        sample_data,
        test_size=0.2,
        random_state=42,
        stratify=sample_data["salary"],
    )

    X_train, y_train, encoder, lb = process_data(
        train_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test_df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, y_train, X_test, y_test, encoder, lb


def test_train_model_returns_expected_model(processed_train_test):
    """
    Ensure train_model returns the expected algorithm type and is fitted.
    """
    X_train, y_train, *_ = processed_train_test
    model = train_model(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    # fitted LogisticRegression should have coef_ after fit
    assert hasattr(model, "coef_")
    assert model.coef_ is not None


def test_inference_output_shape_and_values(processed_train_test):
    """
    Ensure inference returns predictions with correct length and binary values.
    """
    X_train, y_train, X_test, y_test, *_ = processed_train_test
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == y_test.shape[0]

    # predictions should be binary (0/1)
    unique_vals = set(np.unique(preds).tolist())
    assert unique_vals.issubset({0, 1})


def test_compute_model_metrics_range(processed_train_test):
    """
    Ensure precision/recall/F1 are valid probabilities in [0, 1].
    """
    X_train, y_train, X_test, y_test, *_ = processed_train_test
    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    for metric in (precision, recall, fbeta):
        assert isinstance(metric, float)
        assert 0.0 <= metric <= 1.0


def test_save_and_load_model_round_trip(processed_train_test):
    """
    Ensure saving and loading a model preserves inference behavior.
    """
    X_train, y_train, X_test, *_ = processed_train_test
    model = train_model(X_train, y_train)
    preds_before = inference(model, X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "tmp_model.pkl")
        save_model(model, model_path)
        loaded = load_model(model_path)

    preds_after = inference(loaded, X_test)

    assert np.array_equal(preds_before, preds_after)
