# MAO Process Prediction & Trial Analysis

A local application for Micro Arc Oxidation (MAO) historical-trial analysis. It includes process-property prediction, cross-validation, input support warnings, similar-trial retrieval, what-if curves, candidate target search, trial storage, explicit retraining, and PDF reports.

## Start on Windows

1. Install Python 3.11 or 3.12.
2. Double-click `run_mao_app.bat`. On first launch it creates `.venv`, installs dependencies (an internet connection is required), and starts the app.

Or use PowerShell manually from this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If the `py -3.12` launcher is unavailable, use an installed compatible Python version.

## Features

- **Pre-trial prediction:** 11 process-setting inputs.
- **Current-aware analysis:** separate model with 11 process settings plus initial and final current. Intended for post-trial use; calculates `final current − initial current` and displays historical current-change relationships.
- **Outcome predictions:** thickness and Ra minimum/maximum estimates. Training uses midpoint/range models to preserve ordered bounds.
- **Prediction checks:** compares estimates against held-out historical measurements, reports average errors and fit scores, and shows measured-versus-estimated graphs. Missing input values are handled during training; invalid outcome ranges are excluded from fitting and remain visible in the data checks.
- **Historical search and nearest-trial retrieval:** based on normalized numeric inputs and historical records.
- **Data quality:** missing values, endpoint validity checks, and an outlier-review function. Outlier flags never automatically remove records.
- **Extrapolation warnings:** checks individual input ranges and nearest-trial distance. Marginally in-range inputs can still form an unsupported combination.
- **What-if analysis:** varies one parameter while holding the others fixed.
- **Target search:** ranks observed settings and conservative parameter variations. Suggestions are candidates for engineering review, not proven optima.
- **Trial database:** local SQLite storage. Add settings, later enter measured currents/outcomes, and retrain with the recorded trials.
- **PDF report:** inputs, estimated ranges, validation errors, support warnings, closest historical trials, and a plot of the predicted ranges.

## Files and local storage

- `MAO DATASET.xlsx`: original bundled historical workbook; it is not modified by uploads.
- `data/active_dataset.xlsx`: most recently uploaded replacement workbook, if used.
- `data/use_uploaded_dataset.flag`: indicates that the uploaded workbook is selected. Switch back to the original workbook from the sidebar.
- `data/trial_history.sqlite3`: created locally to store new trials.
- `artifacts/mao_model_bundle.joblib`: written when “Retrain and save model artifacts” is used.
- `mao_engine.py`: model training, validation, similarity retrieval, support warnings, and data storage logic.
- `app.py`: user interface, visualizations, and PDF report generation.

The application rebuilds its prediction models from the selected historical workbook and saved measured trial records.

## Important limitations

This is a prototype for data-driven decision support. Cross-validation metrics are preliminary; model selection and reported CV use the same folds, which may make scores optimistic. The displayed 90th-percentile absolute CV error is an empirical error envelope, **not** a calibrated prediction interval. Out-of-range checks do not prove feasibility or safety. Target search does not guarantee an optimal MAO recipe. Validate any candidate through controlled experiments and review with a materials/process engineer before production use.

The prediction output is in micrometres as named in the workbook. The app does not claim that historical correlation establishes process causality.

## Uploading a workbook

Uploading a compatible workbook replaces the active historical workbook; it does not append to the bundled workbook. The original file remains unchanged. Saved trial records in the local SQLite database are merged with whichever historical workbook is active. An uploaded workbook remains selected after a browser refresh or app restart until switched back in the sidebar. The workbook must contain the expected MAO process-setting and measured-outcome column names.
