import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



csv = "/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv"
df  = pd.read_csv(csv)


column_names = df.columns
print(column_names)
#print(df.head())