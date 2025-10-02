import pandas as pd
import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt



#Importing Data
df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataPulled.csv")
print(df.head())

colnames = df.columns.tolist()
print(colnames)

##Data Cleaning / Preprocessing
df_new = df.drop(df.columns[[ 0 ,11, 12, 13]], axis=1)
print(df_new.shape)
print(df_new.columns.tolist())


NAPlayerIDP = df_new['PlayerIDP'].isna().sum()
print(f"IDP Missing: {NAPlayerIDP}")
 
NAPlocID = df_new['PLocID'].isna().sum()
print(f"PLocID Missing: {NAPlocID}")

NAPlayerIDLS = df_new['PlayerIDLS'].isna().sum()
print(f"LSID Missing: {NAPlayerIDLS}")

NASnapLocID = df_new['SnapLocID'].isna().sum()
print(f"SnapLocID Missing: {NASnapLocID}")

NASnapTime = df_new['Snaptime'].isna().sum()
print(f"Snaptime Missing: {NASnapTime}")

NAOP = df_new['OP'].isna().sum()
print(f"OP Missing: {NAOP}")

NAH2F = df_new['H2F'].isna().sum()
print(f"H2F Missing: {NAH2F}")

NAHang = df_new['Hang'].isna().sum()
print(f"Hang Missing: {NAHang}")

NADistance = df_new['Distance'].isna().sum()
print(f"Dist Missing: {NADistance}")

NAPractice = df_new['Practice'].isna().sum()
print(f"Practice Missing: {NAPractice}")

NAGame = df_new['Game'].isna().sum()
print(f"Game Missing: {NAGame}")

NAPrecipitation = df_new['precipitation'].isna().sum()
print(f"Precipitation Missing: {NAPrecipitation}")

NAWind = df_new['Wind'].isna().sum()
print(f"Wind Missing: {NAWind}")

NAGrass = df_new['Grass'].isna().sum()
print(f"Grass Missing: {NAGrass}")

NATurf = df_new['Turf'].isna().sum()
print(f"Turf Missing: {NATurf}")

NATemp = df_new['Temp'].isna().sum()
print(f"Temp Missing: {NATemp}")


PlocID_median = df_new.PLocID.median()
print(f"Punt Loc ID Median: {PlocID_median}")

SnapLocID_median = df_new.SnapLocID.median()
print(f"Snap Loc ID Median: {SnapLocID_median}")

Hang_Median = df_new.Hang.median()
print(f"Hang Median: {Hang_Median}")

Dist_Median = df_new.Distance.median()
print(f"Distance Median: {Dist_Median}")


df_new.Hang = df_new.Hang.fillna(Hang_Median)
df_new.PLocID = df_new.PLocID.fillna(PlocID_median)
df_new.SnapLocID = df_new.SnapLocID.fillna(SnapLocID_median)
df_new.Distance = df_new.Distance.fillna(Dist_Median)

NAPlocID = df_new['PLocID'].isna().sum()
print(f"PLocID Missing New: {NAPlocID}")

NAHang = df_new['Hang'].isna().sum()
print(f"Hang Missing New : {NAHang}")

NADistance = df_new['Distance'].isna().sum()
print(f"Dist Missing New: {NADistance}")

NASnapLocID = df_new['SnapLocID'].isna().sum()
print(f"SnapLocID Missing New: {NASnapLocID}")


X = df_new[['PlayerIDP', 'PLocID', 'PlayerIDLS', 'SnapLocID', 'Snaptime', 'OP', 'H2F', 'Hang', 'Practice', 'Game', 'precipitation', 'Wind', 'Grass', 'Turf', 'Temp']]
y = df_new['Distance']


regr = linear_model.LinearRegression()
regr.fit(X,y)


Coefficients = regr.coef_
Intercept = regr.intercept_

print(f"Formula: {Coefficients} + {Intercept}")

Score = regr.score(X,y)
print(f"Score for all features: {Score}")

#df_new.to_csv('CleanedDistanceData', index=False)


##JobLib Test
import joblib

joblib.dump(regr, 'DistanceLinRegALL')

DL= joblib.load('DistanceLinRegALL')


coefffff = DL.coef_
print(f"TEST:{coefffff}")

