#https://scipy.github.io/devdocs/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

telco = pd.read_csv('Datasets/telco_2023.csv')

data = telco[['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon', 
              'multline', 'voice', 'pager', 'internet', 'forward', 'confer', 'ebill']]

#Using 'ward' method and 'euclidean' metric.
linkage_data = linkage(data, method='ward', metric='euclidean')
dendrogram(linkage_data, color_threshold=450) #Cutting the dendrogram at 450
plt.savefig('Dendrogram - telco.png')
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

telco['cluster'] = hierarchical_cluster.labels_.tolist()
sns.scatterplot(x='longmon', y='equipmon',  hue='cluster', data=telco, color='blue')
plt.show();

print(telco)
telco.to_csv('telco_clusterAssignmentsHierarchical.csv')

df = pd.DataFrame(telco , columns = ["longmon", "tollmon", "equipmon", "cardmon", "wiremon", 
                                     "multline", "voice", "pager", "internet", "forward", "confer", "ebill"])
df['Clusters']=hierarchical_cluster.labels_
print(df)
parallel_coordinates(df, 'Clusters', color=('#383c4a','#0a3661','#dcb536'))
plt.show()