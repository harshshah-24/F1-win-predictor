import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')

class F1RacePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_scaler = None
        self.track_difficulty = self._initialize_track_difficulty()

    def _initialize_track_difficulty(self):
        """Initialize overtaking difficulty for different circuits"""
        # Street circuits and tracks with high overtaking difficulty
        high_difficulty = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Miami', 'Las Vegas']
        # Medium difficulty tracks
        medium_difficulty = ['Hungary', 'Spain', 'Netherlands', 'Australia', 'Imola', 'Qatar']

        return {
            'high': high_difficulty,
            'medium': medium_difficulty
        }

    def load_model(self, filepath):
        """Load a trained model"""
        try:
            # Load the Keras model
            self.model = keras.models.load_model(f"{filepath}_model.h5")

            # Load scalers and other components
            model_components = joblib.load(f"{filepath}_components.pkl")
            self.scaler = model_components['scaler']
            self.target_scaler = model_components['target_scaler']
            self.track_difficulty = model_components['track_difficulty']

            print(f"Model loaded successfully from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_overtaking_difficulty(self, circuit_name):
        """Get overtaking difficulty for a circuit"""
        if circuit_name in self.track_difficulty['high']:
            return 3  # High difficulty
        elif circuit_name in self.track_difficulty['medium']:
            return 2  # Medium difficulty
        else:
            return 1  # Low difficulty

    def is_valid_race_position(self, position):
        """Check if race position is valid (not DNF/R)"""
        if pd.isna(position):
            return False
        if isinstance(position, str):
            return position.upper() != 'R'
        try:
            float(position)
            return True
        except (ValueError, TypeError):
            return False

    def calculate_weighted_mean_position(self, driver_data, current_round, current_year, n_races=5):
        """Calculate weighted mean of last n races for a driver"""
        try:
            # Filter data for the driver up to current round
            driver_history = driver_data[
                (driver_data['Year'] == current_year) &
                (driver_data['Round Number'] < current_round)
            ].sort_values('Round Number', ascending=False)

            if len(driver_history) == 0:
                return 10.0  # Default position for rookies

            # Take last n races (or all available if less than n)
            recent_races = driver_history.head(min(n_races, len(driver_history)))

            if len(recent_races) == 0:
                return 10.0

            # Filter out DNF races and handle missing race positions
            valid_races = recent_races[recent_races['Race Position'].apply(self.is_valid_race_position)]

            if len(valid_races) == 0:
                return 10.0

            # Convert valid positions to float
            positions = []
            for pos in valid_races['Race Position']:
                try:
                    positions.append(float(pos))
                except (ValueError, TypeError):
                    continue

            if len(positions) == 0:
                return 10.0

            positions = np.array(positions)

            # Calculate weights (most recent race gets highest weight)
            n_available = len(positions)
            base_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05][:n_available])
            weights = base_weights / base_weights.sum()  # Normalize weights

            # Check if shapes match before using np.average
            if len(positions) != len(weights):
                print(f"Warning: Position shape {len(positions)} != weights shape {len(weights)}")
                return np.mean(positions)  # Fall back to simple mean

            return np.average(positions, weights=weights)

        except Exception as e:
            print(f"Error in calculate_weighted_mean_position: {e}")
            return 10.0

    def get_team_performance(self, constructors_data, team, current_round, current_year):
        """Get team performance metrics"""
        try:
            team_data = constructors_data[
                (constructors_data['Year'] == current_year) &
                (constructors_data['Round Number'] < current_round) &
                (constructors_data['Team'] == team)
            ]

            if len(team_data) == 0:
                return 5, 0  # Default rank and points for new teams

            # Get latest team performance
            latest_data = team_data.sort_values('Round Number').iloc[-1]
            return latest_data['Rank'], latest_data['Total Points']
        except Exception as e:
            print(f"Error in get_team_performance: {e}")
            return 5, 0

    def get_driver_track_history(self, driver_data, driver_abbr, circuit_name, current_year, years_back=2):
        """Get driver's performance on specific track in last years"""
        try:
            track_history = driver_data[
                (driver_data['Driver Abbr'] == driver_abbr) &
                (driver_data['Circuit Name'] == circuit_name) &
                (driver_data['Year'].isin([current_year - i for i in range(1, years_back + 1)]))
            ].sort_values('Year', ascending=False)

            if len(track_history) == 0:
                return 10.0  # Default if no history

            # Filter out DNF races
            valid_history = track_history[track_history['Race Position'].apply(self.is_valid_race_position)]

            if len(valid_history) == 0:
                return 10.0

            # Convert valid positions to float
            positions = []
            for pos in valid_history['Race Position']:
                try:
                    positions.append(float(pos))
                except (ValueError, TypeError):
                    continue

            if len(positions) == 0:
                return 10.0

            positions = np.array(positions)

            if len(positions) == 1:
                return float(positions[0])
            else:
                # Ensure weights match positions length
                n_positions = len(positions)
                weights = np.array([0.7, 0.3][:n_positions])
                weights = weights / weights.sum()  # Normalize

                if len(positions) != len(weights):
                    return np.mean(positions)

                return np.average(positions, weights=weights)
        except Exception as e:
            print(f"Error in get_driver_track_history: {e}")
            return 10.0

    def get_qualifying_to_race_performance(self, driver_data, driver_abbr, circuit_name, current_year, races_back=3):
        """Get driver's qualifying to race position change on specific track"""
        try:
            track_history = driver_data[
                (driver_data['Driver Abbr'] == driver_abbr) &
                (driver_data['Circuit Name'] == circuit_name) &
                (driver_data['Year'] < current_year)
            ].sort_values('Year', ascending=False).head(races_back)

            if len(track_history) == 0:
                return 0.0  # No change if no history

            # Filter out DNF races and calculate position changes
            changes = []
            for _, row in track_history.iterrows():
                if self.is_valid_race_position(row['Race Position']):
                    try:
                        quali_pos = float(row['Quali Position']) if pd.notna(row['Quali Position']) else 20
                        race_pos = float(row['Race Position'])
                        changes.append(race_pos - quali_pos)
                    except (ValueError, TypeError):
                        continue

            return np.mean(changes) if changes else 0.0
        except Exception as e:
            print(f"Error in get_qualifying_to_race_performance: {e}")
            return 0.0

    def prepare_features(self, drivers_data, constructors_data, prediction_round, circuit_name,
                        starting_grid, mean_temp, rainfall_status, year=2025):
        """Prepare features for prediction"""
        features = []
        driver_abbrs = []

        for grid_pos, driver_abbr in enumerate(starting_grid, 1):
            try:
                # Get driver data
                driver_history = drivers_data[drivers_data['Driver Abbr'] == driver_abbr]

                if len(driver_history) == 0:
                    print(f"Warning: No history found for driver {driver_abbr}")
                    continue

                # Get current team (most recent entry)
                current_team = driver_history.sort_values(['Year', 'Round Number']).iloc[-1]['Team']

                # Feature 1: Weighted mean of last 5 races
                weighted_mean_pos = self.calculate_weighted_mean_position(
                    driver_history, prediction_round, year
                )

                # Feature 2: Team performance
                team_rank, team_points = self.get_team_performance(
                    constructors_data, current_team, prediction_round, year
                )

                # Feature 3: Track temperature and rainfall (encoded)
                temp_normalized = (mean_temp - 20) / 20  # Normalize around typical range
                rain_encoded = 1 if rainfall_status == 'Rain' else 0

                # Feature 4: Overtaking difficulty
                overtaking_diff = self.get_overtaking_difficulty(circuit_name)

                # Feature 5: Driver's track history
                track_history_pos = self.get_driver_track_history(
                    driver_history, driver_abbr, circuit_name, year
                )

                # Feature 6: Qualifying to race performance on this track
                quali_race_change = self.get_qualifying_to_race_performance(
                    driver_history, driver_abbr, circuit_name, year
                )

                # Additional features for deep learning
                grid_position = grid_pos

                # Driver performance variance (consistency metric)
                recent_positions = []
                recent_history = driver_history[
                    (driver_history['Year'] == year) &
                    (driver_history['Round Number'] < prediction_round)
                ].tail(5)

                for _, row in recent_history.iterrows():
                    if self.is_valid_race_position(row['Race Position']):
                        try:
                            recent_positions.append(float(row['Race Position']))
                        except:
                            continue

                performance_variance = np.var(recent_positions) if len(recent_positions) > 1 else 0.0

                feature_row = [
                    float(weighted_mean_pos),
                    float(team_rank),
                    float(team_points),
                    float(temp_normalized),
                    float(rain_encoded),
                    float(overtaking_diff),
                    float(track_history_pos),
                    float(quali_race_change),
                    float(grid_position),
                    float(performance_variance)
                ]

                features.append(feature_row)
                driver_abbrs.append(driver_abbr)

            except Exception as e:
                print(f"Error processing driver {driver_abbr}: {e}")
                continue

        feature_names = [
            'weighted_mean_position', 'team_rank', 'team_points',
            'temperature_normalized', 'rain_status', 'overtaking_difficulty',
            'track_history_position', 'quali_race_change', 'grid_position',
            'performance_variance'
        ]

        return pd.DataFrame(features, columns=feature_names), driver_abbrs

    def predict_race(self, drivers_data, constructors_data, prediction_round, circuit_name,
                    starting_grid, mean_temp, rainfall_status, year=2025):
        """Predict race results"""
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first using load_model()")

        # Prepare features
        features_df, driver_abbrs = self.prepare_features(
            drivers_data, constructors_data, prediction_round, circuit_name,
            starting_grid, mean_temp, rainfall_status, year
        )

        if len(features_df) == 0:
            raise ValueError("No features could be prepared for prediction")

        # Handle NaN values in features
        features_array = features_df.values
        if np.isnan(features_array).any():
            print("Warning: NaN values in prediction features, filling with median")
            features_array = np.nan_to_num(features_array, nan=np.nanmedian(features_array))

        # Scale features
        features_scaled = self.scaler.transform(features_array)

        # Make predictions
        predictions_scaled = self.model.predict(features_scaled, verbose=0)
        predictions = self.target_scaler.inverse_transform(predictions_scaled).flatten()

        # Create results dataframe
        results_df = pd.DataFrame({
            'Driver': driver_abbrs,
            'Predicted_Position': predictions,
            'Starting_Grid': range(1, len(driver_abbrs) + 1)
        })

        # Sort by predicted position
        results_df = results_df.sort_values('Predicted_Position').reset_index(drop=True)
        results_df['Predicted_Rank'] = range(1, len(results_df) + 1)

        print("\n" + "="*50)
        print("RACE PREDICTION RESULTS")
        print("="*50)
        print(results_df[['Predicted_Rank', 'Driver', 'Predicted_Position', 'Starting_Grid']].to_string(index=False))

        return results_df

# Example usage function
def make_prediction(model_path, drivers_csv, constructors_csv, prediction_params):
    """
    Make a race prediction using the trained model

    Parameters:
    - model_path: Path to the saved model (without extension)
    - drivers_csv: Path to drivers data CSV
    - constructors_csv: Path to constructors data CSV
    - prediction_params: Dictionary with prediction parameters
    """
    # Load data
    try:
        drivers_data = pd.read_csv(drivers_csv, encoding='latin-1')
        constructors_data = pd.read_csv(constructors_csv, encoding='latin-1')
        print(f"Loaded drivers data: {drivers_data.shape}")
        print(f"Loaded constructors data: {constructors_data.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Initialize predictor
    predictor = F1RacePredictor()

    # Load the trained model
    if not predictor.load_model(model_path):
        print("Failed to load model")
        return None

    # Make prediction
    try:
        results = predictor.predict_race(
            drivers_data=drivers_data,
            constructors_data=constructors_data,
            prediction_round=prediction_params['round'],
            circuit_name=prediction_params['circuit'],
            starting_grid=prediction_params['grid'],
            mean_temp=prediction_params['temperature'],
            rainfall_status=prediction_params['weather'],
            year=prediction_params['year']
        )
        return results
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None

# Example usage
if __name__ == "__main__":

    # temperature and rainfall for the weekend
    sess = fastf1.get_session(YEAR_GLOBAL, ROUND_GLOBAL, 3)
    sess1 = fastf1.get_session(YEAR_GLOBAL, ROUND_GLOBAL, 4)
    sess.load()
    sess1.load()
    track_temp = (sess.weather_data['TrackTemp'].mean() + sess1.weather_data['TrackTemp'].mean())/2

    # Check if any of the modes for Rainfall is True (indicating rain)
    if sess1.weather_data['Rainfall'].mode().any():
      rainf = 'Rain'
    else:
      rainf = 'Dry'


    # Example prediction parameters
    prediction_params = {
        'round': ROUND_GLOBAL + 1,
        'circuit': 'Silverstone',
        'grid': ["VER", "PIA", "NOR", "RUS", "HAM", "LEC", "ALO", "GAS", "SAI", "ANT", "TSU", "HAD", "ALB", "OCO", "LAW", "BOR", "STR", "BEA", "HUL", "COL"],
        'temperature': track_temp,
        'weather': rainf,
        'year': YEAR_GLOBAL
    }

    # Make prediction (adjust paths as needed)
    results = make_prediction(
        model_path="/content/drive/MyDrive/F1_Data/f1_deep_learning_model",
        drivers_csv="/content/drive/MyDrive/F1_Data/driver_f1new.csv",
        constructors_csv="/content/drive/MyDrive/F1_Data/constructors_f1new.csv",
        prediction_params=prediction_params
    )