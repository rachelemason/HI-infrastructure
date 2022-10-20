#!/usr/bin/env python3
#cnn_support.py
#REM 2022-10-19

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

import os
import glob
import numpy as np
import rasterio
from skimage import measure

DATA_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'

class InstanceSeg():
    """
    Tools for converting from semantic segmentation to instance segmentation
    """

    def __init__(self):
        pass


    def id_instances(self, class_map):
        """
        Convert connected groups of pixels into labelled instances
        """

        instances = measure.label(class_map, connectivity=2)
        return instances


class Ensemble():
    """
    Do ensemble averaging of maps
    """

    def __init__(self, in_path, out_path, apply_to, model_set):
        self.in_path = in_path
        self.out_path = out_path
        self.apply_to = apply_to
        self.model_set = model_set
        self.ensembled_probabilities = {} #defined in self.average_applied_models


    @classmethod
    def open_raster(cls, to_open):
        """
        Given a .tif file, return its first band as an array along with its
        metadata dictionary
        """

        with rasterio.open(to_open, 'r') as f:
            arr = f.read()
            meta = f.meta
        if len(arr.shape) > 2:
            arr = arr[0]

        return arr, meta


    def average_applied_models(self, threshold=0.85, ndvi_threshold=0.5):
        """
        For each test region in self.apply_to, take a simple mean of a set of
        *applied* models, i.e., the *probability* maps that are produced by BFGN.
        Then, convert probabilities to classes using <threshold>, and apply an
        ndvi cut using <ndvi_threshold>. Write each product to file at self.out_path
        and create self.ensembled_probabilities, a dictionary in which
        key=test region name, value = (mean probability map, class map, ndvi_cut_map).
        """

        for region_name, region_data in self.apply_to.items():
            model_list = []

            #get the mean of the relevant models, for each test region
            for applied_model in self.model_set:
                this_file = glob.glob(f'{applied_model}*{region_name}*tif')[0]
                this_array, meta = self.open_raster(this_file)
                model_list.append(this_array)
            mean = np.mean(model_list, axis=0)

            #apply a threshold to convert to binary classes
            classes = np.zeros((mean.shape[0], mean.shape[1]))
            classes[mean == -9999] = -9999
            classes[np.logical_and(mean >= threshold, classes != -9999)] = 1

            #apply the ndvi cut
            ndvi_file = glob.glob(f'{DATA_PATH}{region_data}_ndvi_hires.tif')[0]
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi = f.read()
            cut_classes = np.expand_dims(classes, 0).astype(np.float32)
            cut_classes[ndvi > ndvi_threshold] = 0

            #write all three ensemble products to file
            os.makedirs(self.out_path, exist_ok=True)
            meta.update(count=1)

            for map_type in ['applied_model', 'threshold_model', 'ndvi_cut_model']:
                with rasterio.open(f"{self.out_path}{map_type}.tif", 'w', **meta) as f:
                    f.write(cut_classes)

            self.ensembled_probabilities[region_name] = (mean, classes, cut_classes)
