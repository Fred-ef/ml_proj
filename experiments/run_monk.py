"""Entry point: train and evaluate the NN on MONK 1, 2 and 3.

Produces, for each task, the learning curves (MSE + accuracy) and a summary
table with the MEAN accuracy over several trials/initializations
(GUIDA §1.4, §4 / FAQ). Outputs go to ``results/monk/``.

This is the correctness "collaudo" for the simulator: aim for ~100% accuracy
with a small network (2-4 hidden units). Run this BEFORE the CUP.

To be implemented in F3.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
