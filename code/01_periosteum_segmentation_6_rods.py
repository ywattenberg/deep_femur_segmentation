#!/usr/bin/env python
# coding: utf-8

# [Serena Bonaretti](https://sbonaretti.github.io/)  
# Created: 24 January 2023. Last update: 3 May 2023  
# Narrative license: CC-BY-NC-SA  
# Code license: GNU-GPLv3
# 
# This module calibrates whole femur images acquired with PCCT and segments the periousteum
# --- 


# ------------------------------------------------------------------------------------------------------------------------------------------ 
# IMPORTS

from calibration_phantom import get_rod_coordinates_img
from pre_post_processing import morph_operations_2D

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import csv

import time


def calibrate_and_segment():

    # ------------------------------------------------------------------------------------------------------------------------------------------ 
    # VARIABLES

    # folder and file names

    # to test on server
    image_folder         = "./../../data_in_VM/"
    image_file_name_roots =["1_2208_04835_R_9"]

    for image_file_name_root in image_file_name_roots:

        image_file_name      = image_file_name_root + ".mha"
        rods_file_name       = image_file_name_root + "_rods.csv"
        calibrated_file_name = image_file_name_root + "_calibrated.mha"
        periosteum_file_name = image_file_name_root + "_periosteum.mha"


        # for image processing: 
        # for the parallelization
        n_of_processes = 7

        # n. of phantom rods visible in the image
        n_phantom_rods = 6

        # amount of HA (hydroxyapatite) content in each rod - info from the technical sheet of the phantom
        NOMINAL_VALUES_6_RODS = [797.0, 601.2, 399.6, 201.4, 100.8, 0]

        # threshold for segmentation
        THRESHOLD = 300

        # radius of morphological operators 
        RADIUS = 5


        # ------------------------------------------------------------------------------------------------------------------------------------------ 
        # 1. READING IMAGE

        start = time.time()

        img = sitk.ReadImage(image_folder + image_file_name) 

        end = time.time()
        print ("Computational time: " + str(round(end - start,2)) + "s / " + str(round((end - start)/60,2)) + "min" )

        # ------------------------------------------------------------------------------------------------------------------------------------------ 
        # 2. SEGMENTING THE PHANTOM

        # Identify the calibration phantom, i.e. center and radius of each rod on each slice. 
        calibration_rods = get_rod_coordinates_img (img, n_of_processes)

        # Extract the grey levels from each rod and add them to each sub-list of calibration_rods 

        # for each slice
        for z in range(len(calibration_rods)):

            # initialize container of the instensities of each rod
            all_intensities = []

            # extract the calibration values
            slice_id  = calibration_rods[z][0]
            all_cx    = calibration_rods[z][1]
            all_cy    = calibration_rods[z][2]
            all_radii = calibration_rods[z][3]
            
            # extract the slice from the image
            current_slice = img[:,:,slice_id]
            
            # for each rod
            for cx, cy, radius in zip (all_cx, all_cy, all_radii):

                # initialize rod intensity
                rod_intensities = []
                
                # reduce the radius (to avoid boundary effect) and square it
                radius = radius-int(radius * 0.70)
                radius_squared = radius*radius

                # for each pixel in the rod
                for i in range (cx-radius, cx+radius):
                    for j in range (cy-radius, cy+radius):

                        # calculate the distance from the center and square it
                        dx = i-cx
                        dy = j-cy
                        distance_squared = dx * dx + dy * dy

                        # if the pixel is within the rod circle with reduced radius
                        if distance_squared < radius_squared:
                            # get its intensity and append it to 
                            rod_intensities.append(current_slice[i,j])

                # calculate the average intensity for the rod, round it to the second decimal, and add it to the list
                mean_intensity = round(sum(rod_intensities)/len(rod_intensities),2)
                all_intensities.append(mean_intensity)            
            
            # append to the sublist in calibration_rods
            calibration_rods[z].append(all_intensities)


        # Save the calibration phantom segmentation in a .csv file with the same data structure

        with open(image_folder + rods_file_name, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(calibration_rods)


        # ------------------------------------------------------------------------------------------------------------------------------------------ 
        # 3. CALIBRATING IMAGE

        # Based on the number of rods in the image, determine which values of HA to use
        # select the amount of intensities depending on the number of rods in the image
        if n_phantom_rods == 4:
            nominal_values_rods = NOMINAL_VALUES_6_RODS[1:-1]
        else:
            nominal_values_rods = NOMINAL_VALUES_6_RODS.copy()


        # Calibrate the image

        start = time.time()

        # initialize the calibrated image to float
        img = sitk.Cast(img, sitk.sitkFloat64)

        # for each slice
        for i in range (img.GetSize()[2]): 
            
            # get the rod intensities in the current slice
            current_slice_rod_intensities = calibration_rods[i][4]
            
            # calculate the regression line
            m,b = np.polyfit(current_slice_rod_intensities, nominal_values_rods, 1)
            
            # calibrate the image
            img[:,:,i] = sitk.GetImageFromArray(sitk.GetArrayFromImage(img[:,:,i]) * m + b)

        end = time.time()
        print ("Computational time: " + str(round(end - start,2)) + "s / " + str(round((end - start)/60,2)) + "min" )

        # cast calibrated to integers because otherwise image is too big
        img = sitk.Cast(img, sitk.sitkInt32)
        sitk.WriteImage(img, image_folder + calibrated_file_name)


        # ------------------------------------------------------------------------------------------------------------------------------------------ 
        # 4. THRESHOLDING IMAGE 

        mask = img > THRESHOLD
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        # ------------------------------------------------------------------------------------------------------------------------------------------ 
        # 5. SEGMENTING PERIOUSTEUM 

        # The combination of morphological operators is similar to the one from Burghardt (2010) [1]:
        mask = morph_operations_2D (mask, radius=RADIUS)

        # Labeling the binary blobs and extracting the biggest one (code from [2]): 
        # convert binary image into a connected component image, each component has an integer label.
        mask = sitk.ConnectedComponent(mask)

        # re-label components so that they are sorted according to size (there is an optional minimumObjectSize parameter to get rid of small components).
        mask = sitk.RelabelComponent(mask, sortByObjectSize=True)

        # get largest connected componet, label==1 in sorted component image.
        periosteum = mask == 1

        sitk.WriteImage(periosteum, image_folder + image_file_name_root + "periosteum.mha")



# ------------------------------------------------------------------------------------------------------------------------------------------ 
# 7. References

# [1] Burghardt (2010), www.doi.org/10.1016/j.bone.2010.05.034  
# [2] https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958


 
# ------------------------------------------------------------------------------------------------------------------------------------------ 
#  RUN THE CODE

if __name__ == "__main__":
    start = time.time()

    calibrate_and_segment()

    end = time.time()
    print ("Computational time: " + str(round(end - start,2)) + "s / " + str(round((end - start)/60,2)) + "min" )

    print sitk.__version__
    print np.__version__
    print plt.__version__
    print csv.__version__

