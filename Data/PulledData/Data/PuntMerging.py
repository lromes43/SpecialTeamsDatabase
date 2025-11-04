import pandas as pd

#Goal here is to combine the two csv's and add the final column of view to the raw data 
df1 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Data/PulledData/Data/Punt_ViewPulled.csv")

df2 = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Data/PulledData/Data/PuntANDConditionsPulled.csv")




df_new = pd.merge(df1, df2, how= "outer")
unique_cols = df_new.columns.unique()
print(unique_cols)


df_final = df_new[unique_cols]

print(f"Original columns after merge: {len(df_new.columns)}")
print(f"Unique columns after cleaning: {len(df_final.columns)}")