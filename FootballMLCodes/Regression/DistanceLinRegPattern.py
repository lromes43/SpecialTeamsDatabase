import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/CleanedDistanceDataReg17Features")
print(df.head())
colnames = df.columns.to_list()
print(colnames)




#PlayerIDP vs Distance
plt.scatter(df['PlayerIDP'],df['Distance'], color = 'blue')
plt.xlabel('PlayerIDP')
plt.ylabel('Distance')
#plt.show() ##No relation
X1 = df[['PlayerIDP']]
y = df['Distance']
regr1 = linear_model.LinearRegression()
print(X1.shape)
print(y.shape)
regr1.fit(X1,y)
regr1score = regr1.score(X1,y)
print(f"Player ID vs Distance Score: {regr1score}")


#PlocID vs Distance
plt.scatter(df['PLocID'],df['Distance'], color = 'blue')
plt.xlabel('PLocID')
plt.ylabel('Distance')
#plt.show() ##No Relation
X2 = df[['PLocID']]
y = df['Distance']
regr2 = linear_model.LinearRegression()
regr2.fit(X2,y)
regr2score = regr2.score(X2,y)
print(f"PLocID vs Distance Score: {regr2score}")


#PlayerIDLS vs Distance
plt.scatter(df['PlayerIDLS'],df['Distance'], color = 'blue')
plt.xlabel('PlayerIDLS')
plt.ylabel('Distance')
#plt.show() ##No Relation
X3 = df[['PlayerIDLS']]
y = df['Distance']
regr3 = linear_model.LinearRegression()
regr3.fit(X3,y)
regr3score = regr3.score(X3,y)
print(f"PlayerIDLS vs Distance Score: {regr3score}")


#SnapLocID vs Distance
plt.scatter(df['SnapLocID'],df['Distance'], color = 'blue')
plt.xlabel('SnapLocID')
plt.ylabel('Distance')
#plt.show() ##No Relation
X4 = df[['SnapLocID']]
y = df['Distance']
regr4 = linear_model.LinearRegression()
regr4.fit(X4,y)
regr4score = regr4.score(X4,y)
print(f"SnapLocID vs Distance: {regr4score}")

#Snaptime vs Distance
plt.scatter(df['Snaptime'],df['Distance'], color = 'blue')
plt.xlabel('Snaptime')
plt.ylabel('Distance')
#plt.show() ##relation
X5 = df[['Snaptime']]
y = df['Distance']
regr5 = linear_model.LinearRegression()
regr5.fit(X5,y)
regr5score = regr5.score(X5,y)
print(f"Snaptime vs Distance Score: {regr5score}")


#OP vs Distance
plt.scatter(df['OP'],df['Distance'], color = 'blue')
plt.xlabel('OP')
plt.ylabel('Distance')
#plt.show() ##Relation
X6 = df[['OP']]
y = df['Distance']
regr6 = linear_model.LinearRegression()
regr6.fit(X6,y)
regr6score = regr6.score(X6,y)
print(f"OP vs Distance Score: {regr6score}")


#H2F vs Distance
plt.scatter(df['H2F'],df['Distance'], color = 'blue')
plt.xlabel('H2F')
plt.ylabel('Distance')
#plt.show() ##relation
X7 = df[['H2F']]
y = df['Distance']
regr7 = linear_model.LinearRegression()
regr7.fit(X7,y)
regr7score = regr7.score(X7,y)
print(f"H2F vs Distance Score: {regr7score}")


#Hang vs Distance
plt.scatter(df['Hang'],df['Distance'], color = 'blue')
plt.xlabel('Hang')
plt.ylabel('Distance')
#plt.show() ##relation
X8 = df[['Hang']]
y = df['Distance']
regr8 = linear_model.LinearRegression()
regr8.fit(X8,y)
regr8score = regr8.score(X8,y)
print(f"Hang vs Distance Score: {regr8score}")


#Practice vs Distance
plt.scatter(df['Practice'],df['Distance'], color = 'blue')
plt.xlabel('Practice')
plt.ylabel('Distance')
#plt.show() ##no relation
X9 = df[['Practice']]
y = df['Distance']
regr9 = linear_model.LinearRegression()
regr9.fit(X9,y)
regr9score = regr9.score(X9,y)
print(f"Practice vs Distance Score: {regr9score}")

##Game vs Distance
plt.scatter(df['Game'],df['Distance'], color = 'blue')
plt.xlabel('Game')
plt.ylabel('Distance')
#plt.show() ##maybe relation
X10 = df[['Game']]
y = df['Distance']
regr10 = linear_model.LinearRegression()
regr10.fit(X10,y)
regr10score = regr10.score(X10,y)
print(f"Game vs Distance Score: {regr10score}")

#Precipitation vs Distance
plt.scatter(df['precipitation'],df['Distance'], color = 'blue')
plt.xlabel('precipitation')
plt.ylabel('Distance')
#plt.show() ## maybe relation
X11 = df[['precipitation']]
y = df['Distance']
regr11 = linear_model.LinearRegression()
regr11.fit(X11,y)
regr11score = regr11.score(X11,y)
print(f"Precipitaton vs Distance Score: {regr11score}")

##Wind vs Distance
plt.scatter(df['Wind'],df['Distance'], color = 'blue')
plt.xlabel('Wind')
plt.ylabel('Distance')
#plt.show() #no relation
X12 = df[['Wind']]
y = df['Distance']
regr12 = linear_model.LinearRegression()
regr12.fit(X12,y)
regr12score = regr12.score(X12,y)
print(f"Wind vs Distance Score: {regr12score}")

##Grass vs Distance
plt.scatter(df['Grass'],df['Distance'], color = 'blue')
plt.xlabel('Grass')
plt.ylabel('Distance')
#plt.show() ##no relation
X13 = df[['Grass']]
y = df['Distance']
Regr13 = linear_model.LinearRegression()
Regr13.fit(X13,y)
Regr13Score = Regr13.score(X13,y)
print(f"Grass vs Distance Score: {Regr13Score}")


##Turf vs Distance
plt.scatter(df['Turf'],df['Distance'], color = 'blue')
plt.xlabel('Turf')
plt.ylabel('Distance')
#plt.show() ##no relation
X14 = df[['Turf']]
y = df['Distance']
Regr14 = linear_model.LinearRegression()
Regr14.fit(X14,y)
Regr14Score = Regr14.score(X14,y)
print(f"Turf vs Distance Score: {Regr14Score}")



##Temp vs Distance
plt.scatter(df['Temp'],df['Distance'], color = 'blue')
plt.xlabel('Temp')
plt.ylabel('Distance')
#plt.show() ##maybe relation
X15 = df[['Temp']]
y = df['Distance']
Regr15 = linear_model.LinearRegression()
Regr15.fit(X15,y)
Regr15Score = Regr15.score(X15,y)
print(f"Temp vs Distance Score: {Regr15Score}")

'''



'''
