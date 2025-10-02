from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

df= pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())
print(df.head())


##principal component analysis to reduce dimensions
X = df[['SnapLocID', 'OP' ]]
print(X.head())

pca = PCA(n_components=1)
principal_components = pca.fit_transform(X)

#new df for pca
pcadf = pd.DataFrame(data=principal_components, columns=['PC1'])
pcadf['Distance'] = df['Distance']
print(pcadf)

#intial plotting
plt.scatter(pcadf[['PC1']], pcadf[['Distance']])
plt.xlabel('PC1(SnapLocID, OP)')
plt.ylabel('Distance')
plt.title('Intial Clustering')
plt.show()

#Elbow Method 
sse = []
k_range = range(1,10)
for k in k_range:
    KM = KMeans(n_clusters=k)
    KM.fit(pcadf[['PC1', 'Distance']])
    sse.append(KM.inertia_)

plt.plot(k_range, sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.show() #K = 5 optimal

#Kmeans Implemented
KM = KMeans(n_clusters=5)
y_pred = KM.fit_predict(pcadf[['PC1', 'Distance']])
print(y_pred)

pcadf['cluster'] = y_pred
df1 = pcadf[pcadf.cluster ==0]
df2 = pcadf[pcadf.cluster ==1]
df3 = pcadf[pcadf.cluster ==2]
df4 = pcadf[pcadf.cluster ==3]
df5 = pcadf[pcadf.cluster ==4]

centroids = KM.cluster_centers_





plt.scatter(df1[['PC1']], df1[['Distance']], color = 'pink', label = "Cluster 1")
plt.scatter(df2[['PC1']], df2[['Distance']], color = 'blue', label = "Cluster 2")
plt.scatter(df3[['PC1']], df3[['Distance']], color = 'green', label = "Cluster 3")
plt.scatter(df4[['PC1']], df4[['Distance']], color = 'purple', label = "Cluster 4")
plt.scatter(df5[['PC1']], df5[['Distance']], color = 'brown', label = "Cluster 5")
plt.scatter(centroids[:, 0], centroids[:,1], color = 'black', marker='+',label = 'centroids')
plt.xlabel('PC1: SnapLocID, OP')
plt.ylabel('Distance')
plt.title('SnapLocID and OP vs Hang')
plt.legend()
plt.show()


'''


#Implementing KMeans

KM = KMeans(n_clusters=4)
y_pred = KM.fit_predict(df[['Distance', 'Hang']])
print(y_pred)

df['cluster'] = y_pred

df1 = df[df.cluster ==0]
df2 = df[df.cluster ==1]
df3 = df[df.cluster ==2]
df4 = df[df.cluster ==3]
#df5 = pcadf[pcadf.cluster ==4]
#df6 = pcadf[pcadf.cluster ==5]



centroids = KM.cluster_centers_


plt.scatter(df1[['Distance']], df1[['Hang']], color = 'pink', label = "Cluster 1")
plt.scatter(df2[['Distance']], df2[['Hang']], color = 'blue', label = "Cluster 2")
plt.scatter(df3[['Distance']], df3[['Hang']], color = 'green', label = "Cluster 3")
plt.scatter(df4[['Distance']], df4[['Hang']], color = 'purple', label = "Cluster 4")
#plt.scatter(df5[['PC1']], df5[['PC2']], color = 'brown', label = "Cluster 5")
#plt.scatter(df6[['PC1']], df6[['PC2']], color = 'red', label = "Cluster 6")
plt.scatter(centroids[:, 0], centroids[:,1], color = 'black', marker='+',label = 'centroids')
plt.xlabel('Distance')
plt.ylabel('Hang')
plt.title('Distance vs Hang')
plt.legend()
plt.show()



features_to_average = ['Distance', 'Hang']
cluster_averages = df.groupby('cluster')[features_to_average].mean()


print(f"Cluster Averages: {cluster_averages}")



'''
