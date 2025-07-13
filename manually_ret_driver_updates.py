file_path = '/content/drive/MyDrive/F1_Data/driver_f1new.csv'
df = pd.read_csv(file_path)
df.loc[
    (df['Year'] == YEAR_GLOBAL) &
    (df['Round Number'] == ROUND_GLOBAL) &
    (df['Driver Abbr'] == RET_DRIVER),
    'Race Position'
] = 'R'
df.to_csv(file_path, index=False)