#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

#Loading data
telco = pd.read_csv('Datasets/telco_2023.csv')

print(telco.head())
print(telco.info())

#Selecting features for clustering
data = telco[['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
              'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill']]

#Elbow method
sse = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init='auto')
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11), sse, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia/SSE')
plt.savefig('Elbow method - telco.png')
plt.show()

#Applying k-means clustering with the appropriate number of clusters
k = 5
kmeans = KMeans(n_clusters=k, n_init='auto')
kmeans.fit(data)

print('SSE:', kmeans.inertia_)
print('Final locations of the centroid:', kmeans.cluster_centers_)
print("The number of iterations required to converge", kmeans.n_iter_)

#Assigning cluster labels to each customer
telco['cluster'] = kmeans.labels_.tolist()

#Ploting some clusters in scatter plots
sns.scatterplot(x='longmon', y='tollmon', hue='cluster', data=telco)
plt.show()
sns.scatterplot(x='equipmon', y='cardmon', hue='cluster', data=telco)
plt.show()
sns.scatterplot(x='wiremon', y='ebill', hue='cluster', data=telco)
plt.show()

print(telco)

#Saving cluster assignments to a csv file
telco.to_csv('telco_clusterAssignments.csv')

df = pd.DataFrame(telco ,columns = ['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
                                     'multiline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill'])
df['Clusters']=kmeans.labels_
print(df)
parallel_coordinates(df, 'Clusters', color=('#383c4a','#0a3661','#dcb536'))
plt.show()