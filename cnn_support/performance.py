#!/usr/bin/env python3
#performance.py
#REM 2022-02-20


"""
Code for deriving mapping products from probability maps produced by
run_models.py/RunModels.ipynb, and creating various diagnostic plots.
"""

from collections import defaultdict
import glob
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import rasterio
from rasterio.enums import Resampling


class MapManips():
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root, feature_path, test_sets):
        self.model_output_root = model_output_root
        self.feature_path = feature_path
        self.test_sets = test_sets


    def probabilities_to_classes(self, applied_model, threshold_val=0.85, nodata_val=-9999,\
                                verbose=True):
        """
        Converts the class probabilities in an applied model tif into
        binary classes using a probability threshold.
        """

        if verbose:
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
        if verbose:
            print(f'  - Writing {outfile}')
        with rasterio.open(outfile, 'w', **meta) as f:
            f.write(output)


    def ndvi_cut(self, model_dir, ndvi_threshold, verbose=True):
        """
        Apply an NDVI cut to each map array in model_dir that contains binary classes,
        write to file. Finds the NDVI value for each map pixel, and if it is > ndvi_threshold,
        the class is changed from 1 to 0. Purpose is to to exclude trees that have been
        incorrectly identified as buildings.
        """

        for map_file in glob.glob(f'{model_dir}/*threshold*'):

            #get the name of the test region
            test_region = map_file.split('model_')[-1].replace('.tif', '')

            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()
                meta = f.meta

            #open the corresponding NDVI map
            ndvi_file = f'{self.feature_path}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()

            #set pixels with NDVI > threshold to 0 i.e. not-building; write to file
            map_arr[ndvi_arr > ndvi_threshold] = 0
            outfile = map_file.replace('threshold', 'ndvi_cut')
            if verbose:
                print(f"Writing {outfile}")
            with rasterio.open(f"{outfile}", 'w', **meta) as f:
                f.write(map_arr)


class Evaluate():
    """
    Methods for evaluating model performance and map quality and characteristics
    """

    def __init__(self, model_output_root, feature_path, response_path, test_sets):
        self.model_output_root = model_output_root
        self.feature_path = feature_path
        self.response_path = response_path
        self.test_sets = test_sets


    def _get_map_and_ref_data(self, map_file, test_region):
        """
        Helper method that just reads the map and labelled response
        data for the specified region, clips the edges, and returns
        both as 2D arrays
        """

        #open the map
        with rasterio.open(map_file, 'r') as f:
            map_arr = f.read()

        #open the corresponding reference (labelled buildings) file
        ref_file = f'{self.response_path}{self.test_sets[test_region]}_responses.tif'
        with rasterio.open(ref_file, 'r') as f:
            ref_arr = f.read()

        map_arr = map_arr[0][20:-20, 20:-20]
        ref_arr = ref_arr[0][20:-20, 20:-20]

        return map_arr, ref_arr


    @classmethod
    def _histo(cls, ax, data, bins, xtext, xlims, xlinepos=None, legend=True):
        """
        Helper method for self.ndvi_hist and self.amcu_hist. Puts the data into
        the histogram and does some axis stuff.
        """

        ax.hist(data[0], bins=bins, color='k', histtype='step', density=True,\
                label='All pixels (sample)')
        ax.hist(data[1], bins=bins, color='b', density=True, alpha=0.5,\
                 label='Pixels correctly classified as buildings')
        ax.hist(data[2], bins=bins, color='r', density=True, alpha=0.5,\
                label='Pixels incorrectly classified as buildings')
        if xlinepos is not None:
            ax.axvline(xlinepos, color='k', ls='--')
        if legend:
            ax.legend(loc='upper left')
        ax.set_xlim(xlims)
        ax.set_xlabel(xtext)
        ax.set_ylabel('Frequency??')


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

            #open the map and response files
            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the NDVI file so we can get NDVI values for the candidates
            ndvi_file = f'{self.feature_path}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
            ndvi_arr = ndvi_arr[0][20:-20, 20:-20]

            #collect sample of NDVI from all pixels, for plotting
            all_ndvi.append([x for x in random.sample(ndvi_arr.flatten().tolist(), int(1e5))\
                             if x > -9999])

            #make arrays of candidates that are and aren't buildings, based on labeled refs
            correct = np.where(((ref_arr == 1)  & (map_arr == 1)), ndvi_arr, -9999)
            incorrect = np.where(((ref_arr == 0)  & (map_arr == 1)), ndvi_arr, -9999)

            correct_ndvi.append([x for x in correct.flatten().tolist() if x > -9999])
            incorrect_ndvi.append([x for x in incorrect.flatten().tolist() if x > -9999])

        all_ndvi = [x for y in all_ndvi for x in y]
        correct_ndvi = [x for y in correct_ndvi for x in y]
        incorrect_ndvi = [x for y in incorrect_ndvi for x in y]

        bins = [x * 0.01 for x in range(-50, 100, 1)]
        _, ax = plt.subplots(1, 1, figsize=[6, 6])
        self._histo(ax, [all_ndvi, correct_ndvi, incorrect_ndvi], bins, 'NDVI',\
                    [-0.5, 1.0], ndvi_threshold)


    def amcu_hist(self, model_dir):
        """
        Produces a histogram of aMCU values for NDVI maps of test
        regions, for a single model run. Shows values for all pixels (sample thereof),
        and for pixels correctly and incorrectly classified as buildings.
        """

        print(f'Working on ndvi_cut maps from {model_dir}')

        all_amcu = defaultdict(list)
        correct_amcu = defaultdict(list)
        incorrect_amcu = defaultdict(list)

        #for each test_region map
        for map_file in glob.glob(f'{model_dir}/*ndvi_cut*'):

            #get the name of the test region
            test_region = map_file.split('model_')[-1].replace('.tif', '')

            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the aMCU file, resampling to same resolution as map array
            amcu_file = f'{self.feature_path}{self.test_sets[test_region]}_amcu.tif'
            with rasterio.open(amcu_file, 'r') as f:
                meta = f.meta
                amcu_arr = f.read(out_shape=(meta['count'], map_arr.shape[0], map_arr.shape[1]),\
                                resampling=Resampling.bilinear)

            #for each of the 7 aMCU bands
            for band in range(amcu_arr.shape[0]):

                #collect sample of aMCU from all pixels, for plotting
                all_amcu[band].append([x for x in random.sample(amcu_arr[band].flatten().tolist(),\
                                                          int(1e5)) if x > -9999])

                #make arrays of candidates that are and aren't buildings, based on labeled refs
                correct = np.where(((ref_arr == 1) & (map_arr == 1)), amcu_arr[band], -9999)
                incorrect = np.where(((ref_arr == 0)  & (map_arr == 1)), amcu_arr[band], -9999)

                correct_amcu[band].append([x for x in correct.flatten().tolist() if x > -9999])
                incorrect_amcu[band].append([x for x in incorrect.flatten().tolist() if x > -9999])

        for band in all_amcu.keys():
            all_amcu[band] = [x for y in all_amcu[band] for x in y]
        for band in correct_amcu.keys():
            correct_amcu[band] = [x for y in correct_amcu[band] for x in y]
        for band in incorrect_amcu.keys():
            incorrect_amcu[band] = [x for y in incorrect_amcu[band] for x in y]

        fig, _ = plt.subplots(3, 3, figsize=(10, 10))
        xlims = {0: [-500, 2500], 1: [-500, 3000], 2: [-3000, 2000], 3: [-200, 600],\
                 4: [-200, 600], 5: [-200, 600], 6: [-200, 600]}
        bins=50
        for band, ax in zip(all_amcu.keys(), fig.axes):
            self._histo(ax, [all_amcu[band], correct_amcu[band], incorrect_amcu[band]], bins,\
                        f'aMCU band{band}', xlims[band], legend=False)
        plt.tight_layout()


    def amcu_scatter(self, model_dir, xband, yband, lims, incorrect):
        """
        Intended for producing scatters plot of two aMCU bands, normally 2 and 6.
        """

        fig, _ = plt.subplots(3, 3, figsize=(16, 8))

        #get the list of map files, move KonaMauka to the front so it can be used as a reference
        map_list = glob.glob(f'{model_dir}/*ndvi_cut*')
        map_list.insert(0, map_list.pop(map_list.index\
                                        (f'{model_dir}/ndvi_cut_model_KonaMauka.tif')))

        #for each test_region map
        for map_file, ax in zip(map_list, fig.axes):

            #get the name of the test region
            test_region = map_file.split('model_')[-1].replace('.tif', '')

            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the aMCU file, resampling to same resolution as map array
            amcu_file = f'{self.feature_path}{self.test_sets[test_region]}_amcu.tif'
            with rasterio.open(amcu_file, 'r') as f:
                meta = f.meta
                amcu_arr = f.read(out_shape=(meta['count'], map_arr.shape[0], map_arr.shape[1]),\
                                resampling=Resampling.bilinear)

            #get the aMCU bands that go on the x and y axes
            xdata = amcu_arr[xband]
            ydata = amcu_arr[yband]

            #get x and y data where candidates were correctly identified, add to plot
            correct_x = np.where(((ref_arr == 1) & (map_arr == 1)), xdata, -9999)
            correct_y = np.where(((ref_arr == 1) & (map_arr == 1)), ydata, -9999)
            ax.scatter(correct_x, correct_y, color='b', s=1, alpha=0.5, label='Correct')

            #same for incorrect building candidate pixels, optionally
            if incorrect == 'same':
                incorrect_x = np.where(((ref_arr == 0) & (map_arr == 1)), xdata, -9999)
                incorrect_y = np.where(((ref_arr == 0) & (map_arr == 1)), ydata, -9999)
                ax.scatter(incorrect_x, incorrect_y, color='r', s=1, alpha=0.3, label='Incorrect')

            #if we want to plot KonaMauka incorrect pix instead:
            elif incorrect == 'KonaMauka':
                if test_region == 'KonaMauka':
                    kmx = np.where(((ref_arr == 0) & (map_arr == 1)), xdata, -9999)
                    kmy = np.where(((ref_arr == 0) & (map_arr == 1)), ydata, -9999)
                    sns.kdeplot(x=kmx.flatten(), y=kmy.flatten(), clip=(0, 1000), ax=ax)
                else:
                    ax.scatter(kmx, kmy, color='r', s=1, alpha=0.3, label='Incorrect')
            else:
                pass

            ax.hlines(y=3, xmin=0, xmax=900, color='limegreen', ls='--')
            ax.hlines(y=35, xmin=900, xmax=975, color='limegreen', ls='--')
            ax.vlines(900, ymin=3, ymax=35, color='limegreen', ls='--')
            ax.vlines(975, ymin=0, ymax=35, color='limegreen', ls='--')
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
            ax.set_xlabel(f'aMCU band {xband}')
            ax.set_ylabel(f'aMCU band {yband}')
            ax.set_title(test_region)

        plt.tight_layout()
