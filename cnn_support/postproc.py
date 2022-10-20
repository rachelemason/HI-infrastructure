#!/usr/bin/env python3
#cnn_support.py
#REM 2022-10-20

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.features import rasterize
import fiona
import gdal
from skimage import measure
from sklearn.metrics import classification_report

pd.set_option('display.precision', 2)

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
        self.ensembled_probabilities = {} #will contain maps for probs, classes, cut_classes
        self.ensemble_stats = {} #will contain stats for just one kind of map


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


    #TODO: this is copied from cnn_support.Utils, should probably put that class into
    #separate Utils module
    @classmethod
    def boundary_shp_to_mask(cls, boundary_file, background_file):
        """
        Return a numpy array ('mask') with the same shape as <background_file>,
        in which pixels within all polygons in <boundary_file> have values >=0,
        and pixels outside those polygons have value = np.nan.
        """

        boundary_ds = fiona.open(boundary_file, 'r')
        data = gdal.Open(background_file, gdal.GA_ReadOnly)
        geo = data.GetGeoTransform()
        geo = [geo[1], geo[2], geo[0], geo[4], geo[5], geo[3]]
        background = data.ReadAsArray()
        mask = np.full(background.shape, np.nan)
        for _, polygon in enumerate(boundary_ds):
            rasterize([polygon['geometry']], transform=geo, default_value=0, out=mask)
        return mask


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
            mean = 1 - mean #need to do this in order to have building class=1

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

            plt.imshow(cut_classes[0][20:-20, 20:-20], vmin=0, vmax=1, cmap='bwr')
            plt.show()

            #write all three ensemble products to file
            os.makedirs(self.out_path, exist_ok=True)
            meta.update(count=1)

            for map_type in ['applied_model', 'threshold_model', 'ndvi_cut_model']:
                with rasterio.open(f"{self.out_path}{map_type}.tif", 'w', **meta) as f:
                    f.write(cut_classes) #THIS IS WRONG!!

            self.ensembled_probabilities[region_name] = (mean, classes, cut_classes)


    def performance_stats(self, map_type, trim=20):
        """
        Put performance stats for the building class in one type of map ('threshold'
        or 'ndvi_cut') into self.ensemble_stats, and also display a dataframe showing
        the stats.
        """

        response_path = DATA_PATH.replace('features', 'buildings')
        boundary_path = DATA_PATH.replace('features', 'boundaries')

        for region_name, region_data in self.apply_to.items():
            response_file = f'{response_path}{region_data}_responses.tif'
            boundary_file = f'{boundary_path}{region_data}_boundary.shp'

            if map_type == 'threshold':
                get_stats_for = self.ensembled_probabilities[region_name][1]
            elif map_type == 'ndvi_cut':
                get_stats_for = self.ensembled_probabilities[region_name][2]
            else:
                raise ValueError('map_type must be one of threshold|ndvi_cut')

            #This reproduces code from cnn_support but that's because I want to
            #avoid importing cnn_support so I can remove a number of the packages it
            #depends on from this environment (to be able to install pandas!!)

            #create an array of the same shape as the applied model 'canvas'
            #in which everything outside the training dataset boundary/boundaries is NaN
            mask = self.boundary_shp_to_mask(boundary_file, response_file)
            response_array, _ = self.open_raster(response_file)

            #insert the labelled responses into the array, inside the training boundaries
            #first, trim the edges off everything as they can contain weird stuff
            mask = mask[trim:-trim, trim:-trim]
            response_array = response_array[trim:-trim, trim:-trim]
            get_stats_for = get_stats_for[:, trim:-trim, trim:-trim]
            mask[response_array != 0] = response_array[response_array != 0]

            #flatten to 1D and remove NaNs
            predicted = get_stats_for.flatten()
            expected = mask.flatten()
            predicted = list(predicted[~(np.isnan(expected))])
            expected = list(expected[~(np.isnan(expected))])

            #get performance metrics for the buildings class
            stats = classification_report(expected, predicted, output_dict=True)
            self.ensemble_stats[region_name] = stats['1.0']

        df = pd.DataFrame.from_dict(self.ensemble_stats).T
        df.loc['mean'] = df.mean()
        df['support'] = df['support'].astype(int)
        display(df)
        