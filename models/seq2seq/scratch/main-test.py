import pandas as pd

# create and save dataframe for testing
df = pd.DataFrame({
	'a':[1,2,3],
	'b':[4,5,6],
	'c':[7,8,9]
	})

dest = 'models-vol/test_dataframe.csv'
df.to_csv(dest)
