# README

This is the reposirory for the data and codes for the publication Yáñez-Cuadra V,Moreno M, Ortega-Culaciati F,Donoso F,Báez JC and Tassara A (2023), Mosaicking Andean morphostructure and seismic cycle crustal deformation patterns using GNSS velocities and machine learning. Front. Earth Sci.11:1096238. doi:10.3389/feart.2023.1096238 

The data files are in the data directory with the names pre2014 for the before 2014 dataset and velocidades_sam.txt for the 2018-2021 velocities.

To apply the clustering, first you need to run the preprocess.py script which calculates the strain. By editing the line 31 you can select the pre or post period.

Once the script has finishing running you can run velocity_clustering.py and strian_clustering.py. The results of these scripts are saved in the output directory.  
