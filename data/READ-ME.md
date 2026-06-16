# Data

Place the official dataset files here (download from the Moodle folder
`ML-25-PRJ`, **this year's edition only** — past editions are not valid):

- `ML-CUP25-TR.csv` — CUP training set (500 rows: id, x1..x12, t1..t4)
- `ML-CUP25-TS.csv` — CUP blind test set (1000 rows: id, x1..x12)
- `template-example-with-random-outputs_ML-CUP25-TS.csv` — output format template
- MONK files (from Moodle or UCI https://archive.ics.uci.edu/dataset/70/monk+s+problems):
  - `monks-1.train`, `monks-1.test`
  - `monks-2.train`, `monks-2.test`
  - `monks-3.train`, `monks-3.test`

These files are git-ignored (see `.gitignore`) to keep the repo clean and the
delivery zip small; they are downloaded, not committed.
