# MAO UI Revision Notes

## Changes made from the review

- Replaced the implementation-focused heading with **MAO Process Prediction & Trial Analysis**.
- Removed the top dashboard counters and the legacy model-file explanation.
- Removed optional Groq/AI commentary from the prediction screen and removed its optional dependency note.
- Added collapsed `What does this do?` help panels for the major work areas, plus short explanations for the principal charts.
- Made the new-trial checkboxes and existing-trial update checkboxes respond immediately by placing them outside Streamlit forms. Their corresponding input fields now appear after the checkbox is changed.
- Added a visible table in What-if analysis listing all 11 process settings and their baseline values, showing which setting is varied and which settings are held constant.
- Changed the validation and report tables to show MAO outcome performance without exposing model algorithm names.
- Improved display labels for positive/negative voltage, treatment time, duty cycle and polarity.
- Added workbook upload validation. An uploaded workbook must contain the expected MAO process-setting and measured-outcome columns and at least three valid measured ranges for both thickness and Ra.
- Made uploaded-workbook selection persistent across browser refreshes and app restarts. Uploading a different workbook replaces the active historical workbook; it does not append to the bundled historical workbook. The bundled file is unchanged. Saved records in the local trial database are still included separately.
- Added sidebar controls to switch between the uploaded workbook and the bundled original.

## Verification performed

- Python syntax/compile checks passed.
- Existing engine unit tests passed: 4/4.
- Supplied MAO workbook passed upload validation; a non-Excel/invalid upload was rejected.
- The full app script passed a stubbed render smoke test, and the prediction flow generated valid PDF bytes.
- A native Streamlit server was not launched in the build environment because Streamlit is not installed there. Run the updated app on the target Windows machine for final interactive verification.

## Important behavior

- A new workbook **replaces the active historical workbook**; it is not combined with the bundled workbook.
- Locally saved trial records remain separate and are included with the active historical workbook when the model is prepared.
- Core prediction does not call an LLM.
- The What-if graph is an estimated response across the historical range of the selected setting while other settings remain fixed. It is not a causal experiment or proof of an optimal recipe.
