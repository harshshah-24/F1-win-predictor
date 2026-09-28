# F1 race prediction

This repository contains the maintained **weekly race prediction pipeline**. It updates completed race data, retrains a fixed baseline model when requested, and predicts the next race from published or manually entered qualifying and starting-grid data. The workflow runs in one Colab notebook; a CPU runtime is enough.

## Repository layout

- [`notebooks/F1_Weekly_Baseline.ipynb`](notebooks/F1_Weekly_Baseline.ipynb) — complete update → train/load → predict workflow and its editable configuration cell.
- [`data/driver_f1new_corrected.csv`](data/driver_f1new_corrected.csv) and [`data/constructors_f1new_corrected.csv`](data/constructors_f1new_corrected.csv) — corrected historical **seed data through 2026 round 12**. Later races are not yet in the repository snapshot.

The old TensorFlow scripts and uncorrected 2025 CSVs were removed because they describe a different model and duplicate the weekly notebook's responsibilities. The weekly model is a fixed XGBoost ranking baseline with a separate DNF probability component. It uses pre-race history, driver and constructor form, circuit information, qualifying, and the actual starting grid. Practice features were tested in the research workflow but were not selected for this production baseline.

## First run in Colab

1. Open the notebook in Colab and copy both CSVs into `/content/drive/MyDrive/F1_Data/fastf1_revision/`. Keep their filenames unchanged. You can instead change `DATA_DIR` in the first code cell to another Drive folder containing both files.
2. The first code cell is set to `COMPLETED_YEAR=2026`, `COMPLETED_ROUND=15` (Baku), `UPDATE_WEEKEND=True`, `TRAIN_MODEL=True`, and `RUN_PREDICTION=False`. Run all cells to fetch missing completed rounds 13–15 and train the model. This needs network access to the race-results API; if results are not available there yet, retry the update later.
3. When the next race grid is available, set `UPDATE_WEEKEND=False`, `TRAIN_MODEL=False`, and `RUN_PREDICTION=True`. Use `INPUT_MODE="api"` for published qualifying and grid, or `INPUT_MODE="manual"` to supply them yourself. Run all cells again. The notebook checks that the saved model matches the historical data and installed package versions; if it rejects the checkpoint, set `TRAIN_MODEL=True`.

The notebook saves its model and versioned run folders under `DATA_DIR/weekly_baseline/`. Each run includes the prediction CSV, input snapshots, qualifying record, features, and a manifest. The corrected historical CSVs in `DATA_DIR` are updated by race key, so rerunning an update does **not** duplicate a weekend. The repository's seed CSVs remain unchanged until you explicitly commit a newer snapshot.

## Weekly switches

- **New race completed; update and retrain:** set `COMPLETED_ROUND` to that race, `PREDICT_ROUND` to the next round, `UPDATE_WEEKEND=True`, `TRAIN_MODEL=True`, and `RUN_PREDICTION=False` until prediction inputs are ready.
- **Predict with the saved model:** set `UPDATE_WEEKEND=False`, `TRAIN_MODEL=False`, `RUN_PREDICTION=True`.
- **Recheck a hypothetical grid:** keep the completed round and saved model fixed, select manual mode, and change only the manual grid and any related inputs.

In manual mode, enter `STARTING_GRID` in actual starting order and `QUALIFYING_ORDER` in qualifying classification order **before penalties**. Enter `QUALIFYING_TIMES` when known; otherwise the fitted preprocessing handles missing values. Specify `MANUAL_CIRCUIT_ID` (the historical API circuit ID for an existing track), `MANUAL_EVENT_NAME`, and `TEAM_OVERRIDES` for any driver whose team differs from the latest historical row. A pit-lane starter belongs in `STARTING_GRID` and `PIT_LANE_STARTERS`. Do not mix different seasons' qualifying and grid tables.

API mode checks the published qualifying and grid. If those inputs are delayed, incomplete, or access-limited, it stops; choose manual mode rather than treating qualifying order as the starting grid without checking penalties.

## Model evaluation

Assess the fixed recipe race by race against starting-grid and simple form baselines using only information available before each race. Winner accuracy, podium recall, rank error/correlation, and DNF calibration answer different questions; one good or bad weekend is not enough to change the recipe. The historical README's TensorFlow MAE claim does **not** describe this model and has been removed.

For the pre-race Baku prediction, Russell was correctly picked to win and 2/3 podium drivers were identified. Full-field rank MAE was 5.00 positions, equal to the starting-grid baseline for that race. This is a single-race observation, not a general performance estimate.
