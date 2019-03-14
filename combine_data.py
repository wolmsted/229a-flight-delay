import pandas as pd

data_loc = 'data/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2017_'

feat_columns = ['Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'Distance', 'ArrDel15', 'Cancelled']
output_columns = ['Month', 'DayofMonth', 'DayOfWeek', 'Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'Distance', 'ArrDel15']
curr_df = pd.read_csv(data_loc + '1.csv', usecols=feat_columns)

for i in range(2,13):
	ith_df = pd.read_csv(data_loc + str(i) + '.csv', usecols=feat_columns)
	curr_df = pd.concat([curr_df, ith_df])

curr_df = curr_df[curr_df['Cancelled'] == 0].dropna()
curr_df.to_csv('data/all_samples.csv', index=False, columns=output_columns)


