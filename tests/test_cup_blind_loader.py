"""Validation of the CUP blind CSV before generating submission predictions."""

from pathlib import Path

import numpy as np
import pytest

from src.data.cup import load_cup_blind


OFFICIAL_BLIND_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "cup" / "ML-CUP25-TS.csv"
)


def _row(identifier, features=None):
    if features is None:
        features = range(12)
    return ",".join(map(str, (identifier, *features)))


def test_official_blind_csv_has_expected_shape_and_ids():
    ids, X = load_cup_blind(OFFICIAL_BLIND_CSV)

    assert ids.shape == (1000,)
    assert X.shape == (1000, 12)
    np.testing.assert_array_equal(ids, np.arange(1, 1001))
    assert np.isfinite(X).all()


def test_comment_lines_single_row_and_order_are_preserved(tmp_path):
    path = tmp_path / "single.csv"
    path.write_text("# CUP blind set\n" + _row(27) + "\n", encoding="utf-8")

    ids, X = load_cup_blind(path)
    np.testing.assert_array_equal(ids, [27])
    np.testing.assert_array_equal(X, [list(range(12))])
    assert X.shape == (1, 12)

    path.write_text(_row(27) + "\n" + _row(3) + "\n", encoding="utf-8")
    ids, X = load_cup_blind(path)
    np.testing.assert_array_equal(ids, [27, 3])
    assert X.shape == (2, 12)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# comments only\n", "at least one row"),
        (_row(1, range(11)), "13 columns"),
        (_row(1, range(13)), "13 columns"),
        (_row(1) + "\n" + _row(2, range(11)), "Invalid CUP blind CSV"),
        (_row("1.5"), "exact integers"),
        (_row("nan"), "non-finite"),
        (_row(1, [*range(11), "inf"]), "non-finite"),
        (_row(1) + "\n" + _row(1), "duplicate IDs"),
        (_row(2**53), "exact integers"),
        (_row(1, [*range(11), "text"]), "Invalid CUP blind CSV"),
    ],
)
def test_rejects_invalid_blind_csv(tmp_path, content, message):
    path = tmp_path / "invalid.csv"
    path.write_text(content + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cup_blind(path)
