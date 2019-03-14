import pandas as pd

df = pd.read_csv('data/all_samples.csv')
# delayed_num = df[df['ArrDel15'] == 1].shape[0]
downsampled = df.sample(n=1200000)
downsampled.to_csv('data/regular.csv', index=False)