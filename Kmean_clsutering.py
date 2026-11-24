"""Apprentissage non supérvisé"""
#Algorithme de k-mean clustering
import numpy  as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
X,y=make_blobs(n_samples=100,centers=3,cluster_std=0.5,random_state=0)

#Visualisation
plt.scatter(X[:,0],X[:,1])

"""Modèle et son entrainement"""
modele_non_supervise=KMeans(n_clusters=3)
modele_non_supervise.fit(X)
predictions=modele_non_supervise.predict(X)
plt.scatter(X[:,0],X[:,1],c=predictions)
plt.show()