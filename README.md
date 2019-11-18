# Roofs & Solar panels

This project is aimed at aiding solar power companies to evaluate the viability and production potential of solar panels for a particular home using satellite images. Evaluating the viability of solar panels for a particular home is a labor-intensive process, given that an employee of a solar power company must personally inspect the home and evaluate the roof type and photovoltaic potential of the home. I propose a solution to simplify this process by developing an AI with the ability to pre-screen homes based on roof type and the photovoltaic potential of their geolocation using satellite image data. 


The 'Project' directory contains all of the code and data used for my project proof of concept. It contains the following files and directories:  

* **roof_data directory**: this contains all of the data used to train the roof-type classifier net. This data is taken from Fatemeh Alidoost, Hossein Arefi; “A CNN-Based Approach for Automatic Building Detection and Recognition of Roof Types Using a Single Aerial Image”, PFG – Journal of Photogrammetry, Remote Sensing and Geoinformation Science, December 2018, Volume 86, Issue 5–6, pp 235–248, https://doi.org/10.1007/s41064-018-0060-5).
* **house_train.py**: this file is the CNN training script. It contructs and outputs a Convolutional Neural Network (saved as **cnn1.h5** in the same directory), as well as **accuracy.png**, a graph of the CNN accuracy on the validation and training sets for the given number of training epochs. 
* **solar_data_nasa.csv**: this file contains example data from NASA's POWER database used to create the **clearness.png** and **radiative_flux.png** graphs based on three months of solar radiation data. 
