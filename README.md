# TDI_challenge

This repository contains the code and data I have used for the completion of the The Data Incubator challenge. 

* The code used for my response to Question 1 of the challenge is contained in *TDI_question1.py*.

The 'Project' directory contains all of the code and data used for my project proof of concept. It contains the following files and directories:  

* **roof_data directory**: this contains all of the data used to train the roof-type classifier net. This data is taken from Fatemeh Alidoost, Hossein Arefi; “A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image”, PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science, December 2018, Volume 86, Issue 5–6, pp 235–248, https://doi.org/10.1007/s41064-018-0060-5).
* **house_train.py**: this file is the CNN training script. It contructs and outputs a Convolutional Neural Network (saved as **cnn1.h5** in the same directory), as well as **accuracy.png**, a graph of the CNN accuracy on the validation and training sets for the given number of training epochs. 
* **solar_data_nasa.csv**: this file contains example data from NASA's POWER database used to create the **clearness.png** and **radiative_flux.png** graphs based on three months of solar radiation data. 
