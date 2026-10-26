"""Retrain the selected CUP model and write the blind test output

Steps:
    1. final MEE on training, validation and internal test
    2. retrain the chosen model on all the training data
    3. predict the 1000 blind test patterns
    4. output to <team-name>_ML-CUP25-TS.csv:
         - 4 comment lines (# names / # team nickname / # ML-CUP25 v1 / # date)
         - 1000 rows "id,o1,o2,o3,o4", no spaces after commas, ids 1..1000
    5. output to <team-name>_abstract.txt (~5 lines)
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
