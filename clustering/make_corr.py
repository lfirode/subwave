import numpy as np
import h5py
import scipy.signal as signal   # For signal processing
from scipy.spatial import distance 
from scipy.cluster import hierarchy

# Build the clusters

T=open('list_det','rt').readlines()
nevents = len(T)
threshold = 0.75
threshold = 1.-threshold
nmin=2

L = open('dt_mean','rt').readlines()
A=np.zeros((nevents,nevents))
for i,l in enumerate(L):
     items = l.strip().split()
     i1 = int(items[0])
     i2 = int(items[1])
     c = float(items[2])
     A[i1-1,i2-1] = c
     A[i2-1,i1-1] = c
     
RR = 1-A
np.fill_diagonal(RR,0)
dissimilarity = distance.squareform(RR)
linkage = hierarchy.linkage(dissimilarity, method="centroid")
clusters = hierarchy.fcluster(linkage, threshold, criterion="distance")
unique, counts = np.unique(clusters, return_counts=True)
cl = unique[np.where(counts>nmin)]

# Save results
sname = 'clustering_{0:4.2f}_{1:d}'.format(1-threshold,nmin)
np.savez(sname,name1=cl,name2=clusters,name3=T)
