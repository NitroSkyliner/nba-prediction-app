import pandas as pd


df = pd.read_csv("../data/nba_games.csv")
print(df.shape)
print(df.head())
print(df.info())
print(df['WL'].value_counts())