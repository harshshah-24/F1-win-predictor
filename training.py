import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import matplotlib.pyplot as plt
import pandas as pd
warnings.filterwarnings('ignore')

class F1DeepLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.track_difficulty = self._initialize_track_difficulty()
        self.history = None

    def _initialize_track_difficulty(self):
        """Initialize overtaking difficulty for different circuits"""
        # Street circuits and tracks with high overtaking difficulty
        high_difficulty = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Miami', 'Las Vegas']

        # Medium difficulty tracks
        medium_difficulty = ['Hungary', 'Spain', 'Netherlands', 'Australia', 'Imola', 'Qatar']

        # Default to low difficulty for other tracks
        return {
            'high': high_difficulty,
            'medium': medium_difficulty
        }

    def build_model(self, input_dim):
        """Build the deep learning model architecture"""
        model = keras.Sequential([
            # Input layer with dropout for regularization
            layers.Dense(256, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Hidden layers with residual connections concept
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.15),

            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),

            # Output layer for regression
            layers.Dense(1, activation='linear')
        ])

        # Custom optimizer with learning rate scheduling
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        # Compile model with appropriate loss function for regression
        model.compile(
            optimizer=optimizer,
            loss='huber',  # Huber loss is robust to outliers
            metrics=['mae', 'mse']
        )

        return model

    def get_callbacks(self):
        """Define callbacks for training"""
        callbacks_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate when plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),

            # Model checkpointing
            callbacks.ModelCheckpoint(
                'best_f1_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        return callbacks_list

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

    def calculate_weighted_mean_position(self, driver_data, current_round, current_year, n_races=7):
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
            base_weights = np.array([1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1][:n_available])
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

    def prepare_training_data(self, drivers_data, constructors_data):
        """Prepare training data from historical races"""
        training_features = []
        training_targets = []

        # Check data structure first
        print("Data columns:")
        print("Drivers data:", list(drivers_data.columns))
        print("Constructors data:", list(constructors_data.columns))

        # Group by year, round, and circuit
        race_groups = drivers_data.groupby(['Year', 'Round Number', 'Circuit Name'])

        processed_count = 0
        error_count = 0
        skipped_dnf_count = 0

        for (year, round_num, circuit), race_data in race_groups:
            try:
                # Skip if we don't have enough data for features
                if round_num == 1 and year == drivers_data['Year'].min():
                    continue

                # Check if we have the required columns
                required_cols = ['Quali Position', 'Race Position', 'Driver Abbr',
                               'Mean Track Temperature', 'Rainfall Status']
                missing_cols = [col for col in required_cols if col not in race_data.columns]
                if missing_cols:
                    print(f"Missing columns in race data: {missing_cols}")
                    continue

                # Get starting grid (quali positions) - handle missing qualifying data
                race_data_copy = race_data.copy()
                race_data_copy['Quali Position'] = race_data_copy['Quali Position'].fillna(20)
                race_data_sorted = race_data_copy.sort_values('Quali Position')
                starting_grid = race_data_sorted['Driver Abbr'].tolist()

                # Get race conditions with fallbacks
                try:
                    mean_temp = race_data['Mean Track Temperature'].iloc[0]
                    if pd.isna(mean_temp):
                        mean_temp = 25.0  # Default temperature
                except:
                    mean_temp = 25.0

                try:
                    rainfall_status = race_data['Rainfall Status'].iloc[0]
                    if pd.isna(rainfall_status):
                        rainfall_status = 'Dry'
                except:
                    rainfall_status = 'Dry'

                # Prepare features for this race
                features_df, driver_abbrs = self.prepare_features(
                    drivers_data, constructors_data, round_num, circuit,
                    starting_grid, mean_temp, rainfall_status, year
                )

                if len(features_df) == 0:
                    continue

                # Get actual race results - ONLY for drivers who finished the race
                for i, driver_abbr in enumerate(driver_abbrs):
                    driver_race_data = race_data[race_data['Driver Abbr'] == driver_abbr]
                    if len(driver_race_data) > 0:
                        actual_position = driver_race_data['Race Position'].iloc[0]

                        # Skip if driver didn't finish (DNF/R) or position is invalid
                        if self.is_valid_race_position(actual_position):
                            try:
                                actual_position_float = float(actual_position)
                                training_features.append(features_df.iloc[i].values)
                                training_targets.append(actual_position_float)
                            except (ValueError, TypeError):
                                skipped_dnf_count += 1
                                continue
                        else:
                            skipped_dnf_count += 1

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} races...")

            except Exception as e:
                error_count += 1
                print(f"Error processing race {year}-{round_num}-{circuit}: {e}")
                if error_count > 10:  # Stop if too many errors
                    print("Too many errors, stopping data preparation")
                    break
                continue

        print(f"Successfully processed {processed_count} races with {error_count} errors")
        print(f"Skipped {skipped_dnf_count} DNF entries during training")

        if len(training_features) == 0:
            raise ValueError("No training data could be prepared")

        return np.array(training_features), np.array(training_targets)

    def train_model(self, drivers_data, constructors_data, epochs=200, batch_size=32, validation_split=0.2):
        """Train the deep learning model"""
        print("Preparing training data...")

        # Add data validation
        print(f"Input data shapes - Drivers: {drivers_data.shape}, Constructors: {constructors_data.shape}")

        X, y = self.prepare_training_data(drivers_data, constructors_data)

        if len(X) == 0:
            raise ValueError("No training data available")

        print(f"Training data shape: {X.shape}")
        print(f"Target data shape: {y.shape}")

        # Check for NaN values
        if np.isnan(X).any():
            print("Warning: NaN values found in features, filling with median")
            X = np.nan_to_num(X, nan=np.nanmedian(X))

        if np.isnan(y).any():
            print("Warning: NaN values found in targets")
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Scale targets to improve training stability
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()

        # Build model
        print("Building deep learning model...")
        self.model = self.build_model(X_train_scaled.shape[1])

        # Train model
        print("Training model...")
        self.history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=self.get_callbacks(),
            verbose=1
        )

        # Evaluate model
        print("Evaluating model...")
        y_pred_scaled = self.model.predict(X_test_scaled)
        y_pred = self.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")

        # Feature importance analysis
        self.analyze_feature_importance(X_train_scaled, y_train_scaled)

        return mae, mse, r2

    def analyze_feature_importance(self, X_train, y_train):
        """Analyze feature importance using permutation importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)

        feature_names = [
            'weighted_mean_position', 'team_rank', 'team_points',
            'temperature_normalized', 'rain_status', 'overtaking_difficulty',
            'track_history_position', 'quali_race_change', 'grid_position',
            'performance_variance'
        ]

        # Calculate baseline performance
        baseline_pred = self.model.predict(X_train)
        baseline_mse = mean_squared_error(y_train, baseline_pred)

        importances = []

        for i, feature_name in enumerate(feature_names):
            # Create permuted dataset
            X_permuted = X_train.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])

            # Calculate performance with permuted feature
            permuted_pred = self.model.predict(X_permuted)
            permuted_mse = mean_squared_error(y_train, permuted_pred)

            # Importance is the increase in error
            importance = permuted_mse - baseline_mse
            importances.append(importance)

        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print(importance_df.to_string(index=False))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance (Increase in MSE)')
        plt.title('Feature Importance Analysis')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return

        print("\n" + "="*50)
        print("TRAINING HISTORY VISUALIZATION")
        print("="*50)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].legend()

        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Training MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].legend()

        # MSE
        axes[1, 0].plot(self.history.history['mse'], label='Training MSE')
        axes[1, 0].plot(self.history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Mean Squared Error')
        axes[1, 0].legend()

        # Learning rate (if available)
        if 'lr' in self.history.history:
            axes[1, 1].plot(self.history.history['lr'], label='Learning Rate')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available',
                          ha='center', va='center', transform=axes[1, 1].transAxes)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            print("No model to save")
            return

        print("\n" + "="*50)
        print("SAVING MODEL")
        print("="*50)

        # Save the Keras model
        self.model.save(f"{filepath}_model.h5")

        # Save scalers and other components
        model_components = {
            'scaler': self.scaler,
            'target_scaler': self.target_scaler,
            'track_difficulty': self.track_difficulty,
            'history': self.history.history if self.history else None
        }
        joblib.dump(model_components, f"{filepath}_components.pkl")
        print(f"Model saved to {filepath}_model.h5 and {filepath}_components.pkl")

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

            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")


def validate_data(drivers_data, constructors_data):
    """Validate input data structure"""
    # Check drivers data
    required_driver_cols = ['Year', 'Round Number', 'Circuit Name', 'Driver Abbr', 'Team',
                           'Quali Position', 'Race Position', 'Mean Track Temperature', 'Rainfall Status']

    # Map common column name variations
    column_mapping = {
        'Quali Position': ['Quali Position', 'Qualifying Position', 'Grid Position'],
        'Mean Track Temperature': ['Mean Track Temperature', 'Track Temperature', 'Temperature'],
        'Rainfall Status': ['Rainfall Status', 'Rain Status', 'Weather']
    }

    # Try to standardize column names
    for standard_name, variations in column_mapping.items():
        for variation in variations:
            if variation in drivers_data.columns and standard_name != variation:
                drivers_data = drivers_data.rename(columns={variation: standard_name})

    missing_cols = [col for col in required_driver_cols if col not in drivers_data.columns]
    if missing_cols:
        print(f"Warning: Missing columns in drivers data: {missing_cols}")

    return drivers_data, constructors_data


def train_model():
    """Main training function"""
    # For Google Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        data_path = '/content/drive/MyDrive/F1_Data/'
    except:
        # For local execution
        data_path = './'

    # Load data
    print("Loading data...")
    try:
        drivers_data = pd.read_csv(f'{data_path}driver_f1new.csv', encoding='latin-1')
        constructors_data = pd.read_csv(f'{data_path}constructors_f1new.csv', encoding='latin-1')

        print(f"Successfully loaded drivers data: {drivers_data.shape}")
        print(f"Successfully loaded constructors data: {constructors_data.shape}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check your file paths and ensure the CSV files exist.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Validate and prepare data
    try:
        drivers_data, constructors_data = validate_data(drivers_data, constructors_data)
        print("Data validation completed successfully.")
    except Exception as e:
        print(f"Error during data validation: {e}")
        return

    # Initialize and train the model
    try:
        print("\nInitializing F1 Deep Learning Model...")
        model = F1DeepLearningModel()

        print("Starting model training...")
        mae, mse, r2 = model.train_model(drivers_data, constructors_data, epochs=200, batch_size=32)

        print(f"\nFinal Model Performance:")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R² Score: {r2:.3f}")

        # Plot training history
        model.plot_training_history()

        # Save the trained model
        model.save_model(f"{data_path}f1_deep_learning_model")

        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Model has been saved and is ready for predictions.")
        print("You can now use the prediction script to make race predictions.")

    except Exception as e:
        print(f"Error during model training: {e}")


if __name__ == "__main__":
    train_model()