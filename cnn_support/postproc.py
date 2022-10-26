#!/usr/bin/env python3
#cnn_support.py
#REM 2022-10-25

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

import os
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from sklearn.metrics import classification_report
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import fiona
import gdal

pd.set_option('display.precision', 2)

DATA_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'

#generic function for opening rasters
#TODO: should really go in some utility file
def open_raster(to_open):
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


class Filter():
    """
    Methods related to filtering candidate building pixels based on spectroscopic information
    """

    def __init__(self, in_path, applied_models):
        self.in_path = in_path
        self.applied_models = applied_models


    def _create_hist(self, data, data_type, cut):
        """
        Helper method for self.histo; actually creates the histogram.
        """

        _, ax = plt.subplots(1, 1, figsize=(8, 6))
        for pix, color, label in zip(data, ['k', 'b'],\
                              ['all pixels (sample)', 'pixels classed as "building"']):
            if len(pix) > int(1e6):
                print(f'Sampling 1e6 pixels, which is {1e6/len(pix)*100:.0f}% of available pixels')
                pix = random.sample(pix, int(1e6))

            ax.hist(pix, bins=100, density=True, color=color, lw=1.5, alpha=0.5, label=label)

        if cut is not None:
            plt.axvline(cut)
        plt.legend(loc='upper left')
        plt.xlabel(data_type)
        plt.ylabel('Ill-defined frequency-related y-label...')

        #NOW SAVE THE FIGURE


    def histo(self, data_type='ndvi', band=0, threshold=0.85, cut=0.5):
        """
        Make a plot showing NDVI for (and sample of) all pixels, and pixels classified
        as buildings.
        """

        all_pix = []
        building_pix = []
        for location, alt_name in self.applied_models.items():

            #get building probability map
            f = glob.glob(f'{self.in_path}applied_model_{location}.tif')[0]
            buildings, _ = open_raster(f)

            #get NDVI/aMCU/whatever map, resampling to higher resolution of CNN-based map
            f = glob.glob(f'{DATA_PATH}{alt_name}_{data_type}.tif')[0]
            with rasterio.open(f) as src:
                data = src.read(out_shape=(buildings.shape[0], buildings.shape[1]),\
                                resampling=Resampling.bilinear)

            #trim dodgy edges, flip map so buildings have high probabilities not low ones
            buildings = 1 - buildings[20:-20, 20:-20]
            data = data[band][20:-20, 20:-20]

            #convert pixels in building probability map above <threshold> to NDVI/whatever
            #(so as to plot NDVI/whatever for building pix only), all others to -9999
            arr = np.where((buildings >= threshold), data, -9999)
            building_pix.extend(arr.flatten().tolist())
            all_pix.extend(data.flatten().tolist())

        all_pix = [a for a in all_pix if a > -9999]
        building_pix = [b for b in building_pix if b > -9999]

        if cut is not None:
            print(f'There are {len(building_pix)} candidate building pixels')
            print(f'Applying the cut preserves {len([b for b in building_pix if b < cut])} pixels')

        self._create_hist([all_pix, building_pix], data_type, cut)


class Ensemble():
    """
    Do ensemble averaging of maps
    """

    def __init__(self, in_path, out_path, apply_to, model_set):
        self.in_path = in_path
        self.out_path = out_path
        self.apply_to = apply_to
        self.model_set = model_set


    def create_ensemble(self, start_from, threshold=None, ndvi_threshold=None, show=False):
        """
        Create a dictionary of ensemble maps starting from the maps in self.model_set,
        applied to the regions specified in self.apply_to. Keys are region, values are map arrays.
        Input maps can either be probability maps (start_from='applied_model'), or models that
        have been converted to binary classes and had an NDVI cut applied
        (start_from='ndvi_cut_model'). In the former case, a threshold and NDVI cut will be applied.
        """

        if start_from not in ['applied_model', 'ndvi_cut_model']:
            raise ValueError("<start_from> must be one of 'applied_model'|'ndvi_cut_model'")

        ensemble = {}
        for region_name, region_data in self.apply_to.items():
            print(f'Working on {region_name}')
            model_list = []

            #get the mean of the relevant maps, for each test region
            for model_dir in self.model_set:
                this_file = glob.glob(f'{model_dir}{start_from}_{region_name}.tif')[0]
                this_array, meta = open_raster(this_file)
                model_list.append(this_array)
            mean = np.mean(model_list, axis=0)

            if start_from == 'applied_model':
                #need to do this in order to have building class=1 when starting from prob. maps
                mean = 1 - mean

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
            else:
                cut_classes = np.expand_dims(mean, 0).astype(np.float32)

            if show:
                plt.imshow(cut_classes[0][20:-20, 20:-20])
                plt.show()

            #write the ndvi_cut map to file
            os.makedirs(self.out_path, exist_ok=True)
            meta.update(count=1)
            with rasterio.open(f"{self.out_path}ndvi_cut_model_{region_name}.tif", 'w',\
                               **meta) as f:
                f.write(cut_classes)

            ensemble[region_name] = cut_classes[0]

        return ensemble


    def choose_cut_level(self, ensemble, n_votes):
        """
        For an ensemble created by self.create_ensemble() with <start_from>='ndvi_cut_model',
        return a version converted to binary classes using <n_votes>. The input ensemble will have
        pixel values = 0-1 in steps of 1/n, where n is the number of models that went into the
        ensemble. For example, an ensemble map created from 5 individual models has pixel values
        like this:
        0 - no models assigned class 'building' to this pixel
        0.2 - 1 model assigned class 'building' to this pixel
        0.4 - 2 models assigned class 'building' to this pixel
        0.6 - 3 models assigned class 'building' to this pixel
        0.8 - 4 models assigned class 'building' to this pixel
        1.0 - 5 models assigned class 'building' to this pixel
        Setting <n_votes>=4 will produce a model with binary building/not-building classes such
        that each building will have received a positive classification from 4 input models.
        This method must be used in order to produce a model that is understood by Performance.
        performance_stats.
        """

        binary_maps = {}
        for region in self.apply_to.keys():
            arr = ensemble[region].copy()
            threshold = np.unique(arr)[n_votes]
            arr[arr >= threshold] = 1
            arr[arr < threshold] = 0
            binary_maps[region] = arr

        return binary_maps


class Performance():
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
                this_array, _ = open_raster(this_file)
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
            response_array, _ = open_raster(response_file)

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
            labels = list(df.index)
            ax.scatter(x, df['precision'], color='b', marker='o', s=100)
            ax.scatter(x, df['recall'], color='lightblue', marker='^', s=100)
            ax.scatter(x, df['f1-score'], color='k', marker='s', s=100)
            ax.set_ylim(0, 0.99)
            ax.set_xlabel(labels)

        plt.subplots_adjust(hspace=0)


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
