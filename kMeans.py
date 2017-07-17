# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:06:01 2017

@author: Georgie
"""

"""
We will perform k-means clustering on the iris dataset.
Based on datacamp's unsupervised learning tutorial www.datacamp.com.
"""


#Packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# Import the iris dataset
iris = datasets.load_iris()


# Do a 2D scatter plot of features. There are 4.
plt.figure(1)
feature_1 = 0
feature_2 = 1
x = iris.data[:,feature_1]
y = iris.data[:,feature_2]
plt.scatter(x,y, alpha = 0.5)
plt.xlabel(iris.feature_names[feature_1])
plt.ylabel(iris.feature_names[feature_2])

# Do k-means clustering with 3 clusters (there are three plants)

model = KMeans(n_clusters=3) 
model.fit(iris.data)
labels = model.predict(iris.data)

# Do a 2D scatter plot of features coloured by their labels.
# We'll also add the cluster centroids.
plt.figure(2)
centroids = model.cluster_centers_
feature_1 = 0
feature_2 = 1
centroids_1 = centroids[:,feature_1]
centroids_2 = centroids[:,feature_2]
x = iris.data[:,feature_1]
y = iris.data[:,feature_2]
plt.scatter(x,y, c = labels,alpha = 0.5)
plt.scatter(centroids_1,centroids_2, marker = 'D', s = 50)
plt.xlabel(iris.feature_names[feature_1])
plt.ylabel(iris.feature_names[feature_2])

# We'll first use the species labels to evaluate the quality of the clustering
# We'll make a data frame of the labels we found and the actual species
df = pd.DataFrame({'labels': labels, 'species': iris.target}) 

# Now do a crosstab
print('No preprocessing')
ct = pd.crosstab(df['labels'], df['species']) 
print(ct) 

# We can also look at the inertia:
print(model.inertia_)
    
#Let's look at how this varies as we vary k
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters = k)
    
    # Fit model to samples
    model.fit(iris.data)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.figure(3)
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#As an exercise we can try standarising our dataset, using a pipeline
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3) 
pipeline = make_pipeline(scaler, kmeans) 
pipeline.fit(iris.data)
labels2 = pipeline.predict(iris.data)

df2 = pd.DataFrame({'labels': labels2, 'species': iris.target}) 
ct2 = pd.crosstab(df2['labels'], df2['species']) 
print('Standardised')
print(ct2)

# Made things worse! What about normalizing instead:
    
normalizer = Normalizer()
kmeans = KMeans(n_clusters=3) 
pipeline = make_pipeline(normalizer, kmeans) 
pipeline.fit(iris.data)
labels3 = pipeline.predict(iris.data)

df3 = pd.DataFrame({'labels': labels3, 'species': iris.target}) 
ct3 = pd.crosstab(df3['labels'], df3['species']) 
print('Normalised')
print(ct3)
# A bit better!