def analyze_f1_round(year, round_number):

    print(f"Round {round_number} of {year}")

    try:
        event = fastf1.get_event(year, round_number)
        print(f"\nRace {event.EventName}")
    except Exception as e:
        print(f"Error loading event for {year} Round {round_number}: {e}")
        return

    drivers_data = {}
    team_points = {}

    session_types = {
        'Sprint': {'pos_col': 'Sprint Position', 'points_col': 'Sprint Points'},
        'Qualifying': {'pos_col': 'Quali Position', 'points_col': None},
        'Race': {'pos_col': 'Race Position', 'points_col': 'Race Points'}
    }

    race_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    sprint_points_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

    session_names = [3,4,5]

    for sess_name in session_names:
        try:
            session = fastf1.get_session(year, round_number, sess_name)
            session.load()

            if session.name in session_types:
                session_results = session.results

                for _, driver_row in session_results.iterrows():
                    driver_abbr = driver_row['Abbreviation']
                    team_name = driver_row['TeamName']
                    position = driver_row['Position']
                    points = driver_row['Points'] if 'Points' in driver_row and pd.notna(driver_row['Points']) else 0

                    if driver_abbr not in drivers_data:
                        drivers_data[driver_abbr] = {
                            'Team': team_name,
                            'Sprint Position': None,
                            'Sprint Points': 0,
                            'Quali Position': None,
                            'Race Position': None,
                            'Race Points': 0,
                        }

                    if session.name == 'Sprint':
                        drivers_data[driver_abbr]['Sprint Position'] = position
                        drivers_data[driver_abbr]['Sprint Points'] = points if points > 0 else sprint_points_map.get(position, 0)
                    elif session.name == 'Qualifying':
                        drivers_data[driver_abbr]['Quali Position'] = position
                    elif session.name == 'Race':
                        drivers_data[driver_abbr]['Race Position'] = position
                        drivers_data[driver_abbr]['Race Points'] = points if points > 0 else race_points_map.get(position, 0)

        except Exception as e:
            pass

    for driver_abbr, data in drivers_data.items():
        team = data['Team']
        if team not in team_points:
            team_points[team] = 0

        team_points[team] += data['Race Points']
        team_points[team] += data['Sprint Points']

    sorted_teams = sorted(team_points.items(), key=lambda item: item[1], reverse=True)

    team_pos_map = {team: i + 1 for i, (team, _) in enumerate(sorted_teams)}

    for driver_abbr, data in drivers_data.items():
        team = data['Team']
        data['Team Total Points'] = team_points.get(team, 0)
        data['Team Position (After Race)'] = team_pos_map.get(team, None)

    df = pd.DataFrame.from_dict(drivers_data, orient='index')
    df.index.name = 'Driver Abbr'
    df = df.reset_index()

    df['Race Position Rank'] = df['Race Position'].rank(method='first', na_option='bottom')
    df['Sprint Position Rank'] = df['Sprint Position'].rank(method='first', na_option='bottom')
    df['Quali Position Rank'] = df['Quali Position'].rank(method='first', na_option='bottom')

    df = df.sort_values(
        by=['Race Position Rank', 'Sprint Position Rank', 'Quali Position Rank'],
        ascending=True
    ).drop(columns=['Race Position Rank', 'Sprint Position Rank', 'Quali Position Rank'])

    desired_driver_columns = [
        'Driver Abbr',
        'Sprint Position',
        'Sprint Points',
        'Quali Position',
        'Race Position',
        'Race Points'
    ]
    df = df.reindex(columns=desired_driver_columns)

    df.index = range(1, len(df) + 1)
    df.index.name = 'Rank'

    print(df.to_string())

    team_df = pd.DataFrame(sorted_teams, columns=['Team', 'Total Points'])
    team_df.index = range(1, len(team_df) + 1)
    team_df.index.name = 'Rank'

    print(team_df.head(10).to_string())

analyze_f1_round(YEAR_GLOBAL, ROUND_GLOBAL)
