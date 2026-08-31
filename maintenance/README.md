# Maintenance materials

This folder holds the operational pieces from the original "Gambling Predictor"
project that produces the picks, ratings, and Ravens data this site runs on:

- `code/` — the Python modules that pull game data, compute power ratings and
  win probabilities, grade past picks, and rebuild `data/site_data.json` and
  `data/ravens_full.json`.
- `persistent-state/` — the model's working state between refreshes (open
  predictions, graded history, calibration parameters, Ravens schedule/log).
- `runbook/sports-predictor-app.md` — the operating manual: refresh checklist,
  data-source notes, and the Ravens betting logic.

These are reference/maintenance files only — GitHub Pages serves `index.html`,
`ravens.html`, and `data/*.json` from the repo root, and none of those changed
when this folder was added (the copies in `data/` were already byte-identical
to the ones in this export; `index.html`/`ravens.html` here are current, not
the older ones the export shipped).

To run an actual refresh (new day's picks, grading, Ravens update) you'd need
an environment that can run these Python scripts against live data sources —
this repo doesn't do that on its own.
