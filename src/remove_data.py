import pandas as pd

# load de dataset
df = pd.read_csv("data/raw/dataset.csv")

# Remove the last 10 rows
df = df.iloc[:-10]

# saved the updated dataset
df.to_csv("data/raw/dataset.csv", index=False)

print(" ultimas 10 filas eliinadas y dataset actualizado")
 
	