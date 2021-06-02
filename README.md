# SUBWAVE

Simple subspace detector

This repository contains two folders: clustering and detection.

The clustering folder contains two files: corr_mat.py and make_corr.py

- In corr_mat.py, we compute inter-event correlation coefficients by considering the average correlation coefficient on 3 stations minimum.
- In make_corr.py, we convert these correlation coefficients into a dissimilarity matrix and group the earthquakes into clusters.

The detection folder also contains two files: make_cluster_eigenvectors and make_subspace_detection.py

- In make_clusters.py, we build a hdf5 file for each cluster which groups the singular vectors calculated for each station after alignment of the waveforms.
- In make_subspace_detection.py, we perform the subspace detection after having defined the stations and the clusters used. The singular vectors stored, for each station, in the hdf5 files are then used to form a subspace in which the continuous data are projected. The user defines the dimension (i.e. the number of singular vectors used) for each cluster.
