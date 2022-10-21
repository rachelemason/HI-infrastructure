#!/usr/bin/env python3
#cnn_support.py
#REM 2022-10-21

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from skimage import measure
from sklearn.metrics import classification_report
import rasterio
from rasterio.features import rasterize
import fiona
import gdal


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


    def average_applied_models(self, threshold=0.85, ndvi_threshold=0.5, show=False):
        """
        For each test region in self.apply_to, take a simple mean of a set of
        *applied* models, i.e., the *probability* maps that are produced by BFGN.
        Then, convert probabilities to classes using <threshold>, and apply an
        ndvi cut using <ndvi_threshold>. Write each product to file at self.out_path
        and create self.ensembled_probabilities, a dictionary in which
        key=test region name, value = (mean probability map, class map, ndvi_cut_map).
        """

        ensembled_probabilities = {}
        for region_name, region_data in self.apply_to.items():
            model_list = []

            #get the mean of the relevant models, for each test region
            for model_dir in self.model_set:
                this_file = glob.glob(f'{model_dir}applied_model_{region_name}.tif')[0]
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

            if show:
                plt.imshow(cut_classes[0][20:-20, 20:-20])
                plt.show()

            #write the ndvi_cut map to file
            os.makedirs(self.out_path, exist_ok=True)
            meta.update(count=1)
            with rasterio.open(f"{self.out_path}ndvi_cut_model_{region_name}.tif", 'w',\
                               **meta) as f:
                f.write(cut_classes)

            ensembled_probabilities[region_name] = cut_classes[0]

        return ensembled_probabilities


    def average_cut_classes(self, show=True):
        """
        For each test region in self.apply_to, assign pixels to classes based on the majority
        vote from each of a set of *ndvi_cut* models.
        """

        ensembled_classes = {}
        for region_name in self.apply_to.keys():
            print(f'Working on {region_name}')
            model_list = []

            #get the relevant ndvi cut models, for each test region
            for model_dir in self.model_set:
                this_file = glob.glob(f'{model_dir}ndvi_cut_model_{region_name}*tif')[0]
                this_array, meta = self.open_raster(this_file)
                model_list.append(this_array)

            arr = np.stack(model_list, axis=0)
            print('   - Calculating mode...')
            arr2 = scipy.stats.mode(arr, axis=0).mode[0]

            if show:
                plt.imshow(arr2)
                plt.show()

            #write the ndvi_cut map to file
            os.makedirs(self.out_path, exist_ok=True)
            arr2 = np.expand_dims(arr2, 0).astype(np.float32)
            meta.update(count=1)
            with rasterio.open(f"{self.out_path}ndvi_cut_model_{region_name}.tif", 'w',\
                               **meta) as f:
                f.write(arr2)

            ensembled_classes[region_name] = arr2[0]

        return ensembled_classes


class Performance(Ensemble):
    """
    Methods for assessing model performance/map quality
    """

    def __init__(self, apply_to, model_set, input_dir):
        self.apply_to = apply_to
        self.model_set = model_set
        self.input_dir = input_dir


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


    def get_individual_models(self):
        """
        Return arrays for the models (maps) that were used to create the ensembles,
        so their performance can be compared
        """

        super_dict = {}
        for idx, model_dir in enumerate(self.model_set):
            model_dict = {}
            for region_name in self.apply_to.keys():
                this_file = glob.glob(f'{model_dir}ndvi_cut_model_{region_name}.tif')[0]
                this_array, _ = self.open_raster(this_file)
                model_dict[region_name] = this_array
            super_dict[idx] = model_dict

        return super_dict


    def performance_stats(self, model, trim=20):
        """
        Return a df containing performance stats for the building class in an ensembled
        ndvi_cut map. The edges of the map are trimmed by <trim>
        pixels in order to avoid messed-up edges being included in stats.
        """

        response_path = DATA_PATH.replace('features', 'buildings')
        boundary_path = DATA_PATH.replace('features', 'boundaries')

        stats_dict = {}
        for region_name, region_data in self.apply_to.items():

            response_file = f'{response_path}{region_data}_responses.tif'
            boundary_file = f'{boundary_path}{region_data}_boundary.shp'

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
            #avoid modifying existing dict
            get_stats_for = model[region_name]
            get_stats_for = get_stats_for[trim:-trim, trim:-trim]
            mask[response_array != 0] = response_array[response_array != 0]

            #flatten to 1D and remove NaNs
            predicted = get_stats_for.flatten()
            expected = mask.flatten()
            predicted = list(predicted[~(np.isnan(expected))])
            expected = list(expected[~(np.isnan(expected))])

            #get performance metrics for the buildings class
            stats = classification_report(expected, predicted, output_dict=True)
            stats_dict[region_name] = stats['1.0']

        df = pd.DataFrame.from_dict(stats_dict).T
        df.loc['mean'] = df.mean()
        df['support'] = df['support'].astype(int)

        return df


    def performance_plot(self, stats_list):
        """
        Make a multi-part figure showing precision, recall, f1-score for both
        the ensemble maps in stats_list, and the individual maps in stats_list.
        """

        fig, _ = plt.subplots(4, 2, figsize=(12, 16))

        x = range(len(self.apply_to))
        for df, ax in zip(stats_list, fig.axes):
            df = df.head(-1)
            labels = [l for l in df.index]
            ax.scatter(x, df['precision'], color='b', marker='o', s=100)
            ax.scatter(x, df['recall'], color='lightblue', marker='^', s=100)
            ax.scatter(x, df['f1-score'], color='k', marker='s', s=100)
            ax.set_ylim(0, 0.99)
            ax.set_xlabel(labels)

        plt.subplots_adjust(hspace=0)
