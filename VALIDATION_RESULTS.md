# MAO model validation snapshot

Dataset: `MAO DATASET.xlsx`
Rows: 583

## Data availability

- Rows with all four outcome fields populated: **502**
- Rows with both thickness endpoints: **541**
- Rows with both Ra endpoints: **535**
- Rows with both initial and final current: **407**
- Rows with all 11 process features populated: **560**

Missing values by field:

| Field | Missing rows |
|---|---:|
| KOH (g/l) | 0 |
| Na₃PO₄ (g/l) | 0 |
| Na₂SiO₃ (g/l) | 0 |
| Voltage (V) +ve | 0 |
| Voltage (V) -ve | 2 |
| Frequency (Hz) | 0 |
| Time (min) | 4 |
| polarity(unipolar-1/bipolar-2) | 1 |
| duty cycle +ve (%) | 17 |
| duty cycle -ve (%) | 1 |
| basic duty cycle (%) | 3 |
| initial current | 134 |
| final current | 175 |
| Thickness Min (μm) | 42 |
| Thickness Max (μm) | 42 |
| Ra Value Min (μm) | 47 |
| Ra Value Max (μm) | 48 |

## Selected models and grouped cross-validation metrics

MAE = mean absolute error, in µm; R² is the coefficient of determination. Exact duplicate process configurations are kept in the same validation fold where possible. The model is selected from three candidates using the same grouped folds whose metrics are reported, so the reported scores can be optimistic and are preliminary.

### Pre-trial (11 process settings)

| Outcome | Selected model | Validation rows | MAE min | MAE max | R² min | R² max |
|---|---|---:|---:|---:|---:|---:|
| Thickness | Random Forest | 541 | 7.696 | 9.268 | 0.333 | 0.370 |
| Ra | SVR | 535 | 0.702 | 0.970 | 0.410 | 0.318 |

Candidate comparison:

| Outcome | Candidate | Mean endpoint MAE | Selected |
|---|---|---:|---|
| Ra | SVR | 0.836 | Yes |
| Ra | Random Forest | 0.865 | No |
| Ra | Extra Trees | 0.867 | No |
| Thickness | Random Forest | 8.482 | Yes |
| Thickness | Extra Trees | 8.701 | No |
| Thickness | SVR | 8.803 | No |

### Current-aware (13 inputs)

| Outcome | Selected model | Validation rows | MAE min | MAE max | R² min | R² max |
|---|---|---:|---:|---:|---:|---:|
| Thickness | Random Forest | 391 | 7.655 | 9.187 | 0.478 | 0.492 |
| Ra | SVR | 384 | 0.703 | 0.945 | 0.491 | 0.395 |

Candidate comparison:

| Outcome | Candidate | Mean endpoint MAE | Selected |
|---|---|---:|---|
| Ra | SVR | 0.824 | Yes |
| Ra | Extra Trees | 0.832 | No |
| Ra | Random Forest | 0.851 | No |
| Thickness | Random Forest | 8.421 | Yes |
| Thickness | Extra Trees | 8.471 | No |
| Thickness | SVR | 9.418 | No |

## Interpretation and caveats

- Thickness errors are still several micrometres and the historical dataset includes broad process variation. Do not report predictions as guaranteed values.
- The current-aware model only trains on rows with both current measurements available, plus a valid outcome pair: 391 rows for thickness and 384 for Ra in this workbook.
- The 90th-percentile absolute cross-validation error is an empirical error summary, not a calibrated prediction interval.
- Missing feature values are imputed inside each training fold. Invalid endpoint pairs (negative targets or minimum greater than maximum) are excluded from fitting; they remain inspectable in the source workbook and data-quality UI.
- Outlier detection only flags records for human review. It does not delete them or assert that the measurements are wrong.
- A feature value within its individual historical range may still form a combination outside the joint historical support. Check nearest-trial warnings and validate candidates experimentally.
- Target search ranks candidate settings; it does not establish causal relationships or prove an optimal/safe recipe.
