from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("/Users/lukeromes/Desktop/Personal/Football ML/Football Data/Punt Data/PuntDataFinal.csv")
print(df.columns.to_list())
print(df.head())


##principal component analysis to reduce dimensions
X = df[['Distance', 'Hang', 'PLocID', 'Efficiency']]
print(X.head())

pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

#new df for pca
pcadf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
print(pcadf)

#intial plotting
plt.scatter(pcadf[['PC1']], pcadf[['PC2']])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Intial Clustering')
plt.show()


#Elbow method
sse = []
k_range = range(1,10)
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(pcadf[['PC1', 'PC2']])
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('K vs SSE')
plt.show()


#Implementing KMeans

KM = KMeans(n_clusters=6)
y_pred = KM.fit_predict(pcadf[['PC1', 'PC2']])
print(y_pred)

pcadf['cluster'] = y_pred

df1 = pcadf[pcadf.cluster ==0]
df2 = pcadf[pcadf.cluster ==1]
df3 = pcadf[pcadf.cluster ==2]
df4 = pcadf[pcadf.cluster ==3]
df5 = pcadf[pcadf.cluster ==4]
df6 = pcadf[pcadf.cluster ==5]



centroids = KM.cluster_centers_


plt.scatter(df1[['PC1']], df1[['PC2']], color = 'pink', label = "Cluster 1")
plt.scatter(df2[['PC1']], df2[['PC2']], color = 'blue', label = "Cluster 2")
plt.scatter(df3[['PC1']], df3[['PC2']], color = 'green', label = "Cluster 3")
plt.scatter(df4[['PC1']], df4[['PC2']], color = 'purple', label = "Cluster 4")
plt.scatter(df5[['PC1']], df5[['PC2']], color = 'brown', label = "Cluster 5")
plt.scatter(df6[['PC1']], df6[['PC2']], color = 'red', label = "Cluster 6")
plt.scatter(centroids[:, 0], centroids[:,1], color = 'black', marker='+',label = 'centroids')
plt.xlabel('PC1(Distance, Hang,PlcoID, Efficiency)')
plt.ylabel('PC2')
plt.title('PC1 vs PC2')
plt.legend()
plt.show()

print(pcadf.head(3))