"""Entry point: CUP hyperparameter screening + grid search.

Runs a coarse-then-fine grid search (k-fold CV) over the CUP training set,
comparing 1 vs >=2 hidden layers and any chosen "extra" investigations
(GUIDA §1.3, §3.1). Saves ranked results and significant learning curves to
``results/cup/`` for the report (and appendix).

Keep the internal test set untouched here (it is only for final risk
estimation in cup_final.py).

To be implemented in F5.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
