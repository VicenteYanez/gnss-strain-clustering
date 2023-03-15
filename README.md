# README

Welcome to the data and code repository for our recent publication, "Mosaicking Andean morphostructure and seismic cycle crustal deformation patterns using GNSS velocities and machine learning" (Yáñez-Cuadra et al., 2023), published in Frontiers in Earth Science. The DOI for this publication is 10.3389/feart.2023.1096238.

Our data files are located in the "data" directory, with the "pre2014" dataset containing data from before 2014 and "velocidades_sam.txt" containing data from 2018-2021.

To apply the clustering, please begin by running the "preprocess.py" script, which calculates the strain. You can select the pre- or post-period by editing line 31 of the script.

Once the "preprocess.py" script has finished running, you can run "velocity_clustering.py" and "strain_clustering.py". The results of these scripts will be saved in the "output" directory.

Thank you for your interest in our work. If you have any questions or comments, please do not hesitate to contact us via mail to Vicente Yáñez Cuadra vicenteyanez@proton.me.
