import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
dataset = pd.read_csv("Mall_Customers.csv")
print(dataset.head())
X = dataset.iloc[:,[3,4]].values
WCSS = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters = i,init = 'k-means++',random_state = 42)
    Kmeans.fit(X)
    WCSS.append(Kmeans.inertia_)
plt.figure(figsize = (8,5))
plt.plot(range(1,11),WCSS,marker = 'o')
plt.title('Elbow Method')
plt.xlabel('Number of Customers')
plt.ylabel('WCSS')
plt.show()
Kmeans = KMeans(n_clusters = 5,init = 'k-means++',random_state = 42)
y_Kmeans = Kmeans.fit_predict(X)
score = silhouette_score(X,y_Kmeans)
print("Silhouette Score : ",score)
cluster_names = {
    0 : "Premium Customer",
    1 : "Careful Customer",
    2 : "Impulsive Customer",
    3 : "Budget Customer",
    4 : "Standard Customer"
}
dataset['Cluster'] = y_Kmeans
dataset['Cluster Name'] = dataset['Cluster'].map(cluster_names)
print(dataset.head())
plt.figure(figsize = (10,7))
plt.scatter(X[y_Kmeans == 0,0],X[y_Kmeans == 0,1],s = 100,label = cluster_names[0])
plt.scatter(X[y_Kmeans == 1,0],X[y_Kmeans == 1,1],s = 100,label = cluster_names[1])
plt.scatter(X[y_Kmeans == 2,0],X[y_Kmeans == 2,1],s = 100,label = cluster_names[2])
plt.scatter(X[y_Kmeans == 3,0],X[y_Kmeans == 3,1],s = 100,label = cluster_names[3])
plt.scatter(X[y_Kmeans == 4,0],X[y_Kmeans == 4,1],s = 100,label = cluster_names[4])
plt.scatter(Kmeans.cluster_centers_[:,0],
    Kmeans.cluster_centers_[:,1],
    s = 300,
    label = 'Centroids'
)
plt.title('Customer Segmentation')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()



