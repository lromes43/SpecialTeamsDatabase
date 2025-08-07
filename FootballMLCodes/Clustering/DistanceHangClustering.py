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
X = df[['Distance', 'Hang', ]]
print(X.head())

#pca = PCA(n_components=2)
#principal_components = pca.fit_transform(X)

#new df for pca
#cadf = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
#print(pcadf)

#intial plotting
plt.scatter(df[['Distance']], df[['Hang']])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Intial Clustering')
plt.show()


#Elbow method
sse = []
k_range = range(1,10)
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(df[['Distance', 'Hang']])
    sse.append(km.inertia_)

plt.plot(k_range, sse)
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('K vs SSE')
plt.show()


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


'''
From the graph can see four clusters. These clusters explain the relation between
hang and distance as some punts such as turnover punts result in longer distances
with higher hang times whileas rugby style punts offer great distance but lack hang
further some conditions such as wind require lower kicks edplaining those points and then
there are the mis-hits.
'''


