file_path = '/content/drive/MyDrive/F1_Data/driver_f1new.csv'
df = pd.read_csv(file_path)

year = YEAR_GLOBAL
rnd = ROUND_GLOBAL

try:
  session = fastf1.get_session(year, rnd, 'R')
  session.load()
  results = session.results
  retired = results[results['ClassifiedPosition'] == 'R']

  for _, row in retired.iterrows():
      abbrev = row['Abbreviation']
      updated_rows = df.loc[
          (df['Year'] == year) &
          (df['Round Number'] == rnd) &
          (df['Driver Abbr'] == abbrev),
          'Race Position'
      ].shape[0]
      df.loc[
          (df['Year'] == year) &
          (df['Round Number'] == rnd) &
          (df['Driver Abbr'] == abbrev),
          'Race Position'
      ] = 'R'
except Exception as e:
  print(f"Error: {e}")
df.to_csv(file_path, index=False)