#import packages required for clustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import pca
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import *
from IPython.display import display
import random
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import silhouette_score

white_wine=pd.read_csv('winequality_white.csv')

white_wine.info()

white_wine.describe()

print('Check if any column have missing value', white_wine.isnull().sum())

for i in white_wine.columns:
    plt.figure(figsize=(7,6))
    white_wine[i].hist()
    plt.xlabel(str(i))
    plt.ylabel("freq")

white_wine.drop('free sulfur dioxide',axis=1,inplace=True)

corrmat = white_wine.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaler.fit(white_wine)
X_scaled_array = scaler.transform(white_wine)
X_scaled = pd.DataFrame(X_scaled_array, columns = white_wine.columns)

kmeans=KMeans(n_clusters=2)
kmeans.fit(X_scaled)
#cluster centers
print(kmeans.cluster_centers_)

WCSS = []
##elbow method to know the number of clusters
for i in range(2,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=100,n_init=20,random_state=5)
    kmeans.fit(X_scaled)
    cluster_an=kmeans.predict(X_scaled)
    WCSS.append(kmeans.inertia_)
    plt.scatter(X_scaled_array[:,0],X_scaled_array[:,1],c=cluster_an,s=20)
    centers=kmeans.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1],c='black',s=100,alpha=0.5)
    plt.show()
    s = silhouette_score (X_scaled, cluster_an, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(i, s))
    
plt.plot(range(2,11),WCSS)
plt.title('elbow method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()    




