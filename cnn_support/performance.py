#!/usr/bin/env python3
#performance.py
#REM 2022-02-17


"""
Code for deriving mapping products from probability maps produced by
run_models.py/RunModels.ipynb, and creating various diagnostic plots.
"""

import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import rasterio


class MapManips():
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root):
        self.model_output_root = model_output_root


    def probabilities_to_classes(self, applied_model, threshold_val=0.85, nodata_val=-9999):
        """
        Converts the class probabilities in an applied model tif into
        binary classes using a probability threshold.
        """

        print(f'Converting {applied_model} probabilities to classes')

        with rasterio.open(applied_model) as f:
            arr = f.read()
            meta = f.meta

        output = np.zeros((arr.shape[1], arr.shape[2]))
        output[arr[0] == nodata_val] = nodata_val
        output[np.logical_and(arr[1] >= threshold_val, output != nodata_val)] = 1

        output = np.expand_dims(output, 0).astype(np.float32)
        meta.update({'count': 1})

        outfile = applied_model.replace('applied', 'threshold')
        print(f'  - Writing {outfile}')
        with rasterio.open(outfile, 'w', **meta) as f:
            f.write(output)


class Evaluate():
    """
    Methods for evaluating model performance and map quality and characteristics
    """

    def __init__(self, model_output_root, feature_path, response_path, test_sets):
        self.model_output_root = model_output_root
        self.feature_path = feature_path
        self.response_path = response_path
        self.test_sets = test_sets


    def ndvi_hist(self, model_dir, ndvi_threshold):
        """
        Produces a histogram of NDVI values for threshold maps of test
        regions, for a single model run. Shows values for all pixels (sample thereof),
        and for pixels correctly and incorrectly classified as buildings.
        """

        print(f'Working on threshold maps from {model_dir}')

        all_ndvi = []
        correct_ndvi = []
        incorrect_ndvi = []

        #for each test_region map
        for map_file in glob.glob(f'{model_dir}/*threshold*'):

            #get the name of the test region
            test_region = map_file.split('model_')[-1].replace('.tif', '')

            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()

            #open the labelled response/reference file, which shows where the buildings are
            ref_file = f'{self.response_path}{self.test_sets[test_region]}_responses.tif'
            with rasterio.open(ref_file, 'r') as f:
                ref_arr = f.read()

            #open the NDVI file so we can get NDVI values for the candidates
            ndvi_file = f'{self.feature_path}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
            #collect sample of NDVI from all pixels, for plotting
            all_ndvi.append([x for x in random.sample(ndvi_arr.flatten().tolist(), int(1e5))\
                             if x > -9999])

            #make array containing NDVI in pixels identified as building candidates, else NaN
            candidate_ndvi = np.where((map_arr[0] > 0), ndvi_arr[0], -9999)

            #make arrays of candidates that are and aren't buildings, based on labeled refs
            correct = np.where((ref_arr > 0), candidate_ndvi, -9999)
            incorrect = np.where((ref_arr < 1), candidate_ndvi, -9999)

            correct_ndvi.append([x for x in correct.flatten().tolist() if x > -9999])
            incorrect_ndvi.append([x for x in incorrect.flatten().tolist() if x > -9999])

        all_ndvi = [x for y in all_ndvi for x in y]
        correct_ndvi = [x for y in correct_ndvi for x in y]
        incorrect_ndvi = [x for y in incorrect_ndvi for x in y]

        bins = [x * 0.01 for x in range(-50, 100, 1)]
        plt.hist(all_ndvi, bins=bins, color='0.5', density=False, label='All pixels (sample)')
        plt.hist(correct_ndvi, bins=bins, color='b', density=False, alpha=0.5,\
                 label='Pixels correctly classified as buildings')
        plt.hist(incorrect_ndvi, bins=bins, color='r', density=False, alpha=0.5,\
                label='Pixels incorrectly classified as buildings')
        plt.axvline(ndvi_threshold, color='k', ls='--')
        plt.legend()
        plt.xlabel('NDVI')
        plt.ylabel('Frequency??')
