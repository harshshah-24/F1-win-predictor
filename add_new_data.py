def analyze_f1_round(year, round_number):

    mean_track_temp = 0.0
    rain_chances = 0
    circuit_name = ""
    race_name = ""
    temp_session_count = 0

    try:
        event = fastf1.get_event(year, round_number)
        print(f"{year} Round {round_number}")
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    drivers_data = {}
    team_points = {}

    session_types = {
        'Sprint': {'pos_col': 'Sprint Position', 'points_col': 'Sprint Points'},
        'Qualifying': {'pos_col': 'Quali Position', 'points_col': None},
        'Race': {'pos_col': 'Race Position', 'points_col': 'Race Points'}
    }

    race_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    sprint_points_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

    reference_session = None

    session_names = [3,4,5]

    for sess_name in session_names:
        try:
            session = fastf1.get_session(year, round_number, sess_name)
            session.load()

            circuit_name = session.session_info['Meeting']['Circuit']['ShortName']
            race_name = session.session_info['Meeting']['Name']

            if hasattr(session, 'weather_data') and session.weather_data is not None and not session.weather_data.empty:
                if 'TrackTemp' in session.weather_data.columns:
                    temp_data = session.weather_data['TrackTemp'].dropna()
                    if not temp_data.empty:
                        mean_track_temp += temp_data.mean()
                        temp_session_count += 1

                if 'Rainfall' in session.weather_data.columns:
                    if session.weather_data['Rainfall'].any():
                        rain_chances += 1

            if reference_session is None:
                reference_session = session

            if session.name in session_types or sess_name in ['Q', 'SQ', 'S', 'R']:
                try:
                    session_results = session.results

                    if session_results is not None and not session_results.empty:
                        for _, driver_row in session_results.iterrows():
                            driver_abbr = driver_row.get('Abbreviation', 'UNK')
                            team_name = driver_row.get('TeamName', 'Unknown Team')
                            position = driver_row.get('Position', None)
                            points = driver_row.get('Points', 0) if 'Points' in driver_row and pd.notna(driver_row.get('Points')) else 0

                            if driver_abbr not in drivers_data:
                                try:
                                    driver_number = driver_row.get('DriverNumber', 99)
                                except:
                                    driver_number = 99

                                drivers_data[driver_abbr] = {
                                    'Team': team_name,
                                    'Driver Number': driver_number,
                                    'Sprint Position': None,
                                    'Sprint Points': 0,
                                    'Quali Position': None,
                                    'Race Position': None,
                                    'Race Points': 0,
                                }

                            if sess_name == 'S' or session.name == 'Sprint':
                                drivers_data[driver_abbr]['Sprint Position'] = position
                                drivers_data[driver_abbr]['Sprint Points'] = points if points > 0 else sprint_points_map.get(position, 0)
                            elif sess_name == 'Q' or sess_name == 'SQ' or session.name == 'Qualifying':
                                drivers_data[driver_abbr]['Quali Position'] = position
                            elif sess_name == 'R' or session.name == 'Race':
                                drivers_data[driver_abbr]['Race Position'] = position
                                drivers_data[driver_abbr]['Race Points'] = points if points > 0 else race_points_map.get(position, 0)
                except Exception as e:
                    print(f"Error for {sess_name}: {e}")
                    continue

        except Exception as e:
            continue

    if not drivers_data:
        print(f"No driver data found for {year} Round {round_number}")
        return None, None

    for driver_abbr, data in drivers_data.items():
        team = data['Team']
        if team not in team_points:
            team_points[team] = 0
        team_points[team] += data['Race Points']
        team_points[team] += data['Sprint Points']

    sorted_teams = sorted(team_points.items(), key=lambda item: item[1], reverse=True)

    df = pd.DataFrame.from_dict(drivers_data, orient='index')
    df.index.name = 'Driver Abbr'
    df = df.reset_index()
    df = df.sort_values(by=['Driver Number'], ascending=True)

    final_mean_temp = mean_track_temp / temp_session_count if temp_session_count > 0 else 0
    rainfall_status = "Rain" if rain_chances > 0 else "Dry"

    df.insert(0, 'Year', year)
    df.insert(1, 'Round Number', round_number)
    df.insert(2, 'Circuit Name', circuit_name)
    df.insert(3, 'Race Name', race_name)
    df.insert(4, 'Mean Track Temperature', round(final_mean_temp, 2))
    df.insert(5, 'Rainfall Status', rainfall_status)

    team_df = pd.DataFrame(sorted_teams, columns=['Team', 'Weekend_Points'])
    team_df.insert(0, 'Year', year)
    team_df.insert(1, 'Round Number', round_number)
    team_df.insert(2, 'Circuit Name', circuit_name)
    team_df.insert(3, 'Race Name', race_name)
    team_df.insert(4, 'Mean Track Temperature', round(final_mean_temp, 2))
    team_df.insert(5, 'Rainfall Status', rainfall_status)
    team_df['Rank'] = range(1, len(team_df) + 1)

    return df, team_df

def check_race_exists(year, round_number, existing_df):

    if existing_df.empty:
        return False

    return ((existing_df['Year'] == year) & (existing_df['Round Number'] == round_number)).any()

def append_new_race_data(new_races_list):

    drive_folder = '/content/drive/MyDrive/F1_Data'
    driver_filepath = os.path.join(drive_folder, 'driver_f1new.csv')
    team_filepath = os.path.join(drive_folder, 'constructors_f1new.csv')

    try:
        existing_drivers = pd.read_csv(driver_filepath)
        print(f"Loaded existing driver data: {len(existing_drivers)} records")
    except FileNotFoundError:
        existing_drivers = pd.DataFrame()
        print("No existing driver data found. Creating new file.")

    try:
        existing_teams = pd.read_csv(team_filepath)
        print(f"Loaded existing team data: {len(existing_teams)} records")
    except FileNotFoundError:
        existing_teams = pd.DataFrame()
        print("No existing team data found. Creating new file.")

    new_drivers_data = pd.DataFrame()
    new_teams_data = pd.DataFrame()

    processed_count = 0
    skipped_count = 0

    print(f"\nProcessing {len(new_races_list)} new races")

    for year, round_number in new_races_list:

        try:
            driver_df, team_df = analyze_f1_round(year, round_number)

            if driver_df is not None and team_df is not None:
                new_drivers_data = pd.concat([new_drivers_data, driver_df], ignore_index=True)
                new_teams_data = pd.concat([new_teams_data, team_df], ignore_index=True)

                processed_count += 1
                print(f"{year} R{round_number:02d} - Added {len(driver_df)} drivers, {len(team_df)} teams")
            else:
                print(f"{year} R{round_number:02d} - Not added")

        except Exception as e:
            print(f"{year} R{round_number:02d} - Error: {e}")
            continue

    if not new_drivers_data.empty:
        combined_drivers = pd.concat([existing_drivers, new_drivers_data], ignore_index=True)
        combined_drivers.to_csv(driver_filepath, index=False)
        print(f"\n✓ Updated driver_f1.csv: {len(new_drivers_data)} new records added")
        print(f"  Total driver records: {len(combined_drivers)}")

    if not new_teams_data.empty:
        combined_teams = pd.concat([existing_teams, new_teams_data], ignore_index=True)
        combined_teams.to_csv(team_filepath, index=False)
        print(f"✓ Updated constructors_f1.csv: {len(new_teams_data)} new records added")
        print(f"  Total team records: {len(combined_teams)}")

    # print(f"Processed: {processed_count} new races")
    # print(f"Skipped: {skipped_count} existing races")
    # print(f"Files updated in: {drive_folder}")

def append_single_race(year, round_number):

    append_new_race_data([(year, round_number)])

def append_multiple_races(races_dict):

    races_list = []
    for year, rounds in races_dict.items():
        for round_num in rounds:
            races_list.append((year, round_num))

    append_new_race_data(races_list)

append_single_race(YEAR_GLOBAL, ROUND_GLOBAL)

# append_new_race_data([(2025, 12), (2025, 13), (2025, 14)])

# new_races = {
#     2025: [15, 16, 17],  # Remaining 2025 races
#     2026: [1, 2, 3, 4, 5]  # First few 2026 races
# }
# append_multiple_races(new_races)

# remaining_2025_races = [(2025, i) for i in range(1, 11)]  # Rounds 11-24
# append_new_race_data(remaining_2025_races)