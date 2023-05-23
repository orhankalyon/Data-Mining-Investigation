#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

telco = pd.read_csv('Datasets/telco_2023.csv')

data = telco[['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
              'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill']]

#search for the eps parameter value
from sklearn.neighbors import NearestNeighbors # importing the library
neighb = NearestNeighbors(n_neighbors=2) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(data) # fitting the data to the object
distances,indices=nbrs.kneighbors(data) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.plot(distances) # plotting the distances
plt.show() # showing the plot

neighb = NearestNeighbors(n_neighbors=5) # creating an object of the NearestNeighbors class
nbrs=neighb.fit(data) # fitting the data to the object
distances,indices=nbrs.kneighbors(data) # finding the nearest neighbours

# Sort and plot the distances results
distances = np.sort(distances, axis = 0) # sorting the distances
distances = distances[:, 1] # taking the second column of the sorted distances
plt.plot(distances) # plotting the distances
plt.savefig('K-dist Graph - telco.png')
plt.show() # showing the plot

#parameter eps=18, min_samples=5
dbscan = DBSCAN(eps = 18, min_samples = 5)
dbscan.fit(data)

telco['cluster'] = dbscan.labels_.tolist()

sns.scatterplot(x='longmon', y='equipmon',  hue='cluster', data=telco, color='blue')
plt.show();

print(telco)
telco.to_csv('telco_clusterAssignmentsDBScan.csv')

df = pd.DataFrame(telco ,columns = ['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
                                    'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill'])
df['Clusters']=dbscan.labels_
print(df)
parallel_coordinates(df, 'Clusters',color=('red','blue','green', "yellow", "black"))
plt.show()