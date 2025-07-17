# F1 Project

This project provides a set of tools and scripts for analyzing, updating, and predicting Formula 1 race results using historical data and deep learning models. The scripts are designed to work with data stored in CSV files (drivers and constructors) and leverage the FastF1 library, pandas, and TensorFlow/Keras for data processing and machine learning. Given the qualifying grid and the race circuit, the model predicts the final positions of all 20 drivers.

## Folder Structure

- `driver_team_table.py` — Analyzes a specific F1 round and prints a table of driver and team results for that round.
- `add_new_data.py` — Adds new race data (drivers and teams) to the dataset by fetching from FastF1 and processes weather, points, and session results.
- `team_cumulative_points.py` — Calculates and updates the cumulative constructor points and ranks for each team across all rounds and years.
- `ret_driver_updates.py` — Automatically updates the driver data CSV to mark drivers as retired ('R') for a given year and round using FastF1 session results.
- `manually_ret_driver_updates.py` — Manually marks a specific driver as retired ('R') for a given year and round in the driver data CSV.
- `training.py` — Trains a deep learning model to predict race results using historical driver and constructor data. Handles feature engineering, model training, evaluation, and saving.
- `prediction.py` — Loads the trained model and makes race result predictions for a given round, using the latest data and race conditions.

## Data Files

- `driver_f1new.csv` — Main driver data file (should be located at `/content/drive/MyDrive/F1_Data/driver_f1new.csv`).
- `constructors_f1new.csv` — Main constructor/team data file (should be located at `/content/drive/MyDrive/F1_Data/constructors_f1new.csv`).

## Usage

### 1. Analyzing Driver and Team Results

To print a table of driver and team results for a round:

```python
# Set YEAR_GLOBAL and ROUND_GLOBAL in driver_team_table.py
YEAR_GLOBAL = 2025           # e.g. current season
ROUND_GLOBAL = 12            # e.g. upcoming or recent round
python driver_team_table.py
```

### 2. Adding New Race Data

To add new race data (drivers and teams) for a specific year and round:

```python
# In add_new_data.py, set YEAR_GLOBAL and ROUND_GLOBAL, then run:
YEAR_GLOBAL = 2025           # e.g. current season
ROUND_GLOBAL = 12            # e.g. upcoming or recent round
python add_new_data.py
```

This will fetch and append new data to the CSV files.

### 3. Calculating Team Cumulative Points

To update the constructor CSV with cumulative points and ranks:

```python
python team_cumulative_points.py
```

### 4. Marking Retired Drivers

- **Automatic (using FastF1 results):**

  - Edit `YEAR_GLOBAL` and `ROUND_GLOBAL` in `ret_driver_updates.py` and run:
    
    ```python
    YEAR_GLOBAL = 2025           # e.g. current season
    ROUND_GLOBAL = 12            # e.g. upcoming or recent round
    python ret_driver_updates.py
    ```
- **Manual (for a specific driver):**
  - Set `YEAR_GLOBAL`, `ROUND_GLOBAL`, and `RET_DRIVER` in `manually_ret_driver_updates.py` and run:
    
    ```python
    YEAR_GLOBAL = 2025           # e.g. current season
    ROUND_GLOBAL = 12            # e.g. upcoming or recent round
    python manually_ret_driver_updates.py
    ```

### 5. Training the Deep Learning Model

To train the model on the available data:

```python
python training.py
```

This will save the trained model and scalers to the data folder.

### 6. Making Predictions

To predict race results for a future round:

- Edit the parameters in `prediction.py` (such as `prediction_params` and file paths).
- Give inputs for the starting grid (ie. the qualifying results) and circuit name.
- Run:
  
  ```python
  python prediction.py
  ```

## Features Used in the Training Model

The deep learning model is trained using a set of engineered features that capture both driver and team performance, as well as race and track conditions. The main features are:

1. **Weighted Mean Position**: Weighted average of a driver's finishing positions in the last several races (recent races are weighted more heavily).
2. **Team Rank**: The constructor's (team's) rank in the championship prior to the race.
3. **Team Points**: The constructor's cumulative points prior to the race.
4. **Temperature Normalized**: The mean track temperature for the race weekend, normalized to a typical range.
5. **Rain Status**: Encoded as 1 if rain was present during the weekend, 0 otherwise.
6. **Overtaking Difficulty**: Encoded value (1=low, 2=medium, 3=high) based on the circuit's overtaking difficulty.
7. **Track History Position**: Weighted mean of the driver's finishing positions at the same circuit in previous years.
8. **Qualifying to Race Change**: Average change in position from qualifying to race finish for the driver at the same circuit in previous years.
9. **Grid Position**: The driver's starting grid position for the race.
10. **Performance Variance**: Variance in the driver's finishing positions in recent races (a measure of consistency).

These features are automatically extracted and engineered from the historical data for each driver and team, and are used as input to the deep learning model for both training and prediction.

## Final Model Performance
After training the deep learning model on the engineered features and historical race data, the final model achieved the following evaluation metrics on the test set:

- Mean Absolute Error (MAE): 2.63
- Mean Squared Error (MSE): 13.15
- R² Score: 0.513

These metrics indicate the average prediction error and the proportion of variance in actual race results explained by the model.

## Notes

- All scripts expect the data files to be present in `/content/drive/MyDrive/F1_Data/` (Google Colab path). Adjust paths if running locally.
- The deep learning model uses a variety of engineered features, including recent performance, team strength, weather, and track characteristics.
- The project is modular: you can update data, retrain, and predict independently.

## License

This project is for educational and research purposes. Attribution to the original authors and FastF1 is appreciated. 
