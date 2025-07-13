df = pd.read_csv('/content/drive/MyDrive/F1_Data/constructors_f1new.csv')
df_new = df.copy()

df_new['Team_Cons'] = df_new['Team']

df_new = df_new.sort_values(['Year', 'Round Number']).reset_index(drop=True)

def calculate_cumulative_constructor_points(df):
    all_rows = []

    for year in df['Year'].unique():
        year_data = df[df['Year'] == year].copy()

        team_cumulative = {}

        for round_num in sorted(year_data['Round Number'].unique()):
            round_data = year_data[year_data['Round Number'] == round_num].copy()

            for idx, row in round_data.iterrows():
                team = row['Team']
                weekend_points = row['Weekend_Points']

                if team not in team_cumulative:
                    team_cumulative[team] = 0

                team_cumulative[team] += weekend_points

            for idx, row in round_data.iterrows():
                team = row['Team']
                round_data.loc[idx, 'Total Points'] = team_cumulative[team]

            round_data = round_data.sort_values('Total Points', ascending=False).reset_index(drop=True)
            round_data['Rank'] = range(1, len(round_data) + 1)

            all_rows.append(round_data)

    result_df = pd.concat(all_rows, ignore_index=True)
    return result_df

df_processed = calculate_cumulative_constructor_points(df_new)

original_path = '/content/drive/MyDrive/F1_Data/constructors_f1new.csv'
df_processed.to_csv(original_path, index=False)