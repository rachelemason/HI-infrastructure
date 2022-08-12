#!/usr/bin/env python3
#cnn_support.py
#REM 2022-08-11

"""
Contains functions to help with:
- setting up and visualizing training data with the BFGN package
- looping over training datasets and parameter combinations and
  visualizing results
"""

import os
import shutil
import subprocess
import warnings
import ast
from IPython.utils import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import gridspec
import gdal
import fiona
import rasterio
from sklearn.metrics import classification_report

#BFGN relies on an old version of tensorflow that results in various
#messages and FutureWarnings. Can't do much about that, so they
#are suppressed below.

#first, disable pylint messages caused by calling warnings before imports completed
#pylint: disable=wrong-import-position

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import tensorflow as tf #import only needed for suppressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.logging.set_verbosity(tf.logging.ERROR)

import bfgn.reporting.reports
from bfgn.configuration import configs
from bfgn.data_management import data_core, apply_model_to_data
from bfgn.experiments import experiments


def input_files_from_config(config_file, print_files=True):
    """
    Return a dictionary containing the lists of feature, response, and boundary
    files specified in <config_file>. Dictionary keys are 'feature_files',
    'response_files', and 'boundary_files'
    """

    filedict = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            for kind in ['feature_files:', 'response_files:', 'boundary_files:']:
                if kind in line:
                    line = line.replace(kind, '').strip()
                    files = ast.literal_eval(line)
                    filedict[kind] = files
                    if print_files:
                        print(f'{kind} {files}')

    return filedict['feature_files:'], filedict['response_files:'], filedict['boundary_files:']


def rasterize(reference_raster, shapefile_list, replace_existing):
    """
    Converts the .shp files in <shapefile_list> into .tif files, using
    the geometry info in <reference_raster>. Output tif files are written
    to the same directory as the input shapefiles.
    """

    feature_set = gdal.Open(reference_raster, gdal.GA_ReadOnly)
    trans = feature_set.GetGeoTransform()

    for file in shapefile_list:
        if not replace_existing:
            if os.path.isfile(file.replace('shp', 'tif')):
                print(f"{file.replace('shp', 'tif')} already exists, not recreating")
                continue
        print(f'Rasterizing {file} using {reference_raster} as reference')
        cmd_str = 'gdal_rasterize ' + file + ' ' + os.path.splitext(file)[0] +\
                '.tif -init 0 -burn 1 -te ' + str(trans[0]) + ' ' + str(trans[3] +\
                trans[5]*feature_set.RasterYSize) + ' ' + str(trans[0] +\
                trans[1]*feature_set.RasterXSize) + ' ' + str(trans[3]) + ' -tr ' +\
                str(trans[1]) + ' ' + str(trans[5])
        subprocess.call(cmd_str, shell=True)


def boundary_shp_to_mask(boundary_file, background_file):
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
        rasterio.features.rasterize([polygon['geometry']], transform=geo,\
                                    default_value=0, out=mask)
    return mask


def _hillshade(ax, to_plot):
    """
    Helper function for show_input_data() etc. Plots data on an axis using
    hillshading, using default parameters for sun azimuth and altitude.
    """

    light = LightSource(azdeg=315, altdeg=45)
    ax.imshow(light.hillshade(to_plot), cmap='gray')


def show_input_data(feature_file, response_file, boundary_file, hillshade=True):
    """
    Creates a 2-panel plot to show (1) the whole 'training canvas'; the area containing the
    training data and to which the model will eventually be applied, and
    (2) the labelled features (responses) within their boundary region(s). Code adapted from
    the ecoCNN tutorial.
    """

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # show the whole map to which the model will eventually be applied
    map_tif = gdal.Open(feature_file, gdal.GA_ReadOnly)
    map_array = map_tif.ReadAsArray()
    map_array[map_array == map_tif.GetRasterBand(1).GetNoDataValue()] = np.nan

    if hillshade:
        _hillshade(ax1, map_array)
    else:
        ax1.imshow(map_array)

    # show the boundary of the region(s) containing the training data
    mask = boundary_shp_to_mask(boundary_file, feature_file)

    # show where the labelled features are, within the training boundaries
    responses = gdal.Open(response_file, gdal.GA_ReadOnly).ReadAsArray()
    mask[responses != 0] = responses[responses != 0] # put the buildings on the canvas
    ax2.imshow(mask)

    plt.tight_layout()


def plot_training_data(features, responses, images_to_plot=3, feature_band=0, nodata_value=-9999):
    """ Tool to plot the training and response data data side by side. Adapted from the ecoCNN code.
        Arguments:
        features - 4d numpy array
        Array of data features, arranged as n,y,x,p, where n is the number of samples, y is the
        data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius),
        and p is the number of features.
        responses - 4d numpy array
        Array of of data responses, arranged as n,y,x,p, where n is the number of samples, y is the
        data y dimension (2*window_size_radius), x is the data x dimension (2*window_size_radius),
        and p is the response dimension (always 1).
    """

    features = features.copy()
    responses = responses.copy()
    features[features == nodata_value] = np.nan
    responses[responses == nodata_value] = np.nan

    feat_nan = np.squeeze(np.isnan(features[:,:,:,0]))
    features[feat_nan,:] = np.nan

    _ = plt.figure(figsize=(4,images_to_plot*2))
    gs1 = gridspec.GridSpec(images_to_plot, 2)
    for n in range(0, images_to_plot):
        _ = plt.subplot(gs1[n,0])

        feat_min = np.nanmin(features[n,:,:,feature_band])
        feat_max = np.nanmax(features[n,:,:,feature_band])

        plt.imshow(features[n,:,:,feature_band], vmin=feat_min, vmax=feat_max)
        plt.xticks([])
        plt.yticks([])
        if n == 0:
            plt.title('Feature')

        _ = plt.subplot(gs1[n,1])
        plt.imshow(responses[n,:,:,0], vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        if n==0:
            plt.title('Response')


def probabilities_to_classes(method, applied_model_arr, threshold_val=0.95, nodata_value=-9999):
    """
    Converts the class probabilities in the applied_model array (not tif) into binary classes
    using maximum likelihood or a threshold.
    """

    output = np.zeros((applied_model_arr.shape[1], applied_model_arr.shape[2]))
    output[applied_model_arr[0] == nodata_value] = nodata_value

    if method == 'ML':
        output[output != nodata_value] =\
                      np.argmax(applied_model_arr, axis=0)[output != nodata_value]
    elif method == 'threshold':
        output[np.logical_and(applied_model_arr[1] >=\
                              threshold_val, output != nodata_value)] = 1
    else:
        print("<threshold> parameter must be one of ['ML', 'threshold']")

    return output


def tif_to_array(tif):
    """
    Returns contents of <tif> as a numpy array. Some BFGN methods, like
    data_management.apply_model_to_data.apply_model_to_site, write a tif
    instead of returning an array, so this methid is useful for further
    operations on those files (but can be used for any tif)
    """

    data = gdal.Open(tif, gdal.GA_ReadOnly)
    arr = data.ReadAsArray()
    arr[arr == data.GetRasterBand(1).GetNoDataValue()] = np.nan

    return arr


def show_applied_model(applied_model, original_img, zoom, responses=None, hillshade=True,\
                      filename=None):
    """
    Plots the applied model created by bfgn.data_management.apply_model_to_data.apply_model_to_site,
    converted to a numpy array by self.applied_model_as_array. Also zooms into a subregion and
    shows (1) the applied model probabilities, (2) the applied model converted to classes, and (3)
    the original image.
    """

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

    if responses is not None:
        shape = gdal.Open(responses, gdal.GA_ReadOnly)
        shape = shape.ReadAsArray()
        shape = np.ma.masked_where(shape < 1.0, shape)

    #Show the applied model probabilities for class 1 over the whole area to which it
    #has been applied
    ax1.imshow(applied_model[0], cmap='Greys_r')
    if responses is not None:
        ax1.imshow(shape, alpha=0.5, cmap='Reds_r')

    #Zoom into the applied model probabilities
    ax2.imshow(applied_model[0][zoom[0]:zoom[1], zoom[2]:zoom[3]], cmap='Greys_r')

    #Same but with probabilities converted to binary classes using a threshold
    classes = probabilities_to_classes('threshold', applied_model)
    ax3.imshow(classes[zoom[0]:zoom[1], zoom[2]:zoom[3]], cmap='Greys')
    #Overplot responses (buildings), if available
    if responses is not None:
        ax3.imshow(shape[zoom[0]:zoom[1], zoom[2]:zoom[3]], alpha=0.8, cmap='viridis_r')

    #The original image for which everything was predicted (same subset/zoom region)
    original = tif_to_array(original_img)
    original[original < 0.5] = np.nan
    original = original[zoom[0]:zoom[1], zoom[2]:zoom[3]]

    if hillshade:
        _hillshade(ax4, original)
    else:
        ax4.imshow(original)

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=400)


def performance_metrics(applied_model, responses, boundary_file):
    """
    Given a model prediction array and a set of responses, calculate precision,
    recall, and f1-score for each class (currently assumes binary classes)
    """

    # convert class probabilities to actual classes
    classes = probabilities_to_classes('ML', applied_model)

    # create an array of the same shape as the applied model 'canvas'
    # in which everything outside the training dataset boundary/boundaries is NaN
    mask = boundary_shp_to_mask(boundary_file, responses)

    # insert the labelled responses into the array, inside the training boundaries
    response_array = tif_to_array(responses)
    mask[response_array != 0] = response_array[response_array != 0]

    # flatten to 1D and remove NaNs
    predicted = classes.flatten()
    expected = mask.flatten()
    predicted = list(predicted[~(np.isnan(expected))])
    expected = list(expected[~(np.isnan(expected))])

    # get performance metrics
    print('Calculating metrics...')
    stats = classification_report(expected, predicted, output_dict=True)

    return stats


class Loops():
    """
    Methods for looping over BFGN parameter combinations and evaluating results.
    """

   #using settatr in __init__ is nice but causes pylint to barf, so
   #pylint: disable=no-member

    def __init__(self, application_data, iteration_data):
        for key in application_data:
            setattr(self, key, application_data[key])
        for key in iteration_data:
            setattr(self, key, iteration_data[key])
        self.default_out = None #defined in loop_over_configs()
        self.stats_array = None #defined in loop_over_configs()


    def _archive_and_tidy(self, num):
        """
        Helper method for loop_over_configs(). Archives model reports and settings files, and also
        removes now-redundant model and data directories.
        """

        #archive the model report and settings file
        shutil.move(f'{self.default_out}model_report.pdf',\
                    f'{self.out_path}combo_{num}/model_report.pdf')
        shutil.move(f'new_settings_{num}.yaml',\
                    f'{self.out_path}combo_{num}/new_settings_{num}.yaml')

        #delete any existing munged data and model output
        shutil.rmtree('./data_out', ignore_errors=True)
        shutil.rmtree(self.default_out, ignore_errors=True)


    def loop_over_configs(self):
        """
        Loops over BFGN configurations (can be training data or other parameters in the
        config file), and returns a numpy array of model performance metrics (precision,
        recall, F1-score) for all combinations.
        This function intentionally produces much less diagnostic info than if the model
        were fit outside it, using direct calls to BFGN; looping over parameters assumes
        the user already knows what they're doing.
        """

        #Array to hold the performance metrics
        self.stats_array = np.zeros((len(self.parameter_combos),\
                                     len(self.app_features)*3))

        #Loop over the parameter combos, fit model  record metrics
        for idx, params in enumerate(self.parameter_combos):

            #get the current set of parameters
            combo_dict = {}
            for i, j in enumerate(params):
                name = list(self.permutations.keys())[i]
                combo_dict[name] = j

            print('\n===================================================')
            print(f'Testing parameter combination #{idx}: {combo_dict}\n')

            #create a new settings file with these parameters
            #(could probably find a way of not re-opening the settings file every time)
            newlines = []
            with open('settings.yaml', 'r', encoding='utf-8') as f:
                for line in f:
                    #identify and edit lines containing parameters to be changed
                    if any(ele in line for ele in self.permutations.keys()):
                        item = line.split(':')[0].strip()
                        try:
                            newline = f'  {item}: {combo_dict[item]}\n'
                            newlines.append(newline)
                        except KeyError as err:
                            #probably means there is something wrong with the input permutations
                            print(f'WARNING: not changing parameter {err} \n')
                            newlines.append(line)
                    #find where the model output is going to go; will end up being 2nd instance
                    #of 'dir_out' in the config file, which is what we want
                    elif 'dir_out:' in line:
                        self.default_out = line.split(':')[1].strip()+'/'
                        newlines.append(line)
                    else:
                        newlines.append(line)

            #create a new settings file and output dir for this parameter combo
            with open(f'new_settings_{idx}.yaml', 'w', encoding='utf-8') as f:
                for line in newlines:
                    f.write(line)
            os.makedirs(f'{self.out_path}combo_{idx}', exist_ok=True)

            #delete existing munged data and model output to avoid errors
            #TODO: replace hardcoded './data_out' with value read from config file,
            #like for model output dir
            shutil.rmtree('./data_out', ignore_errors=True)
            shutil.rmtree(self.default_out, ignore_errors=True)

            #fit the model using this parameter combo, suppressing the very verbose output
            with io.capture_output():
                config = configs.create_config_from_file(f'new_settings_{idx}.yaml')
                data_container = data_core.DataContainer(config)
                data_container.build_or_load_rawfile_data(rebuild=True)
                data_container.build_or_load_scalers()
                data_container.load_sequences()
                experiment = experiments.Experiment(config)
                experiment.build_or_load_model(data_container=data_container)
                experiment.fit_model_with_data_container(data_container, resume_training=True)
                final_report = bfgn.reporting.reports.Reporter(data_container, experiment, config)
                final_report.create_model_report()

                #for each test region, apply the model and get stats
                for i, _f in enumerate(self.app_features):
                    #apply the model
                    apply_model_to_data.apply_model_to_site(experiment.model, data_container, _f,
                                                        self.default_out+self.app_outnames[i])
                    #convert tif to array
                    applied_model = tif_to_array(self.default_out+self.app_outnames[i]+'.tif')
                    #archive the applied model tif
                    shutil.move(f'{self.default_out+self.app_outnames[i]}.tif',\
                                f"{self.out_path}combo_{idx}/{self.app_outnames[i]}.tif")
                    #make pdf showing applied model
                    show_applied_model(applied_model, zoom=self.zooms[i], original_img=_f[0],\
                                       responses=self.app_responses[i], filename=\
                                       f"{self.out_path}combo_{idx}/{self.app_outnames[i]}\
                                       .pdf")
                    #get performance metrics
                    stats = performance_metrics(applied_model, self.app_responses[i],\
                                                self.app_boundaries[i])

                    #stats for class 1 (building)
                    self.stats_array[idx, i*3] = np.round(stats['1.0']['precision'], 2)
                    self.stats_array[idx, i*3+1] = np.round(stats['1.0']['recall'], 2)
                    self.stats_array[idx, i*3+2] = np.round(stats['1.0']['f1-score'], 2)

            #TODO: record/return number of training samples for future use/plots
            n_samples = np.sum([len(n) for n in data_container.features])
            print(f'{n_samples} training samples were extracted from the data\n')

            #archive model report and config file for this parameter combo/setting
            self._archive_and_tidy(idx)

            #close all figures in the hope of not creating memory problems
            plt.close('all')


    def parameter_heatmap(self):
        """
        Plots a heatmap of precision, recall and F1 score for a set of model test regions and
        parameter combinations, using <stats_array> produced by the loop_over_configs function.
        """

        print(f'Parameters tested:{list(self.permutations.keys())}')

        _, ax = plt.subplots(figsize=(10, 10))
        img = ax.imshow(self.stats_array, vmin=0, vmax=1, cmap='hot')
        _ = plt.colorbar(img, shrink=0.55)

        #add labels to the plot
        #the basic x labels - metrics used
        xlabels = [['precision', 'recall', 'f1-score'] * len(self.app_outnames)][0]

        #ylabels - parameter combos, or nicknames for them
        if self.nicknames is not None:
            ylabels = self.nicknames
        else:
            ylabels = self.parameter_combos

        _ = ax.set_xticks(np.arange(len(xlabels)), labels=xlabels)
        _ = ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)

        #add labels for the test regions the stats refer to
        test_regions = [a.split('_model_')[1] for a in self.app_outnames]
        for n, region in enumerate(test_regions):
            ax.text(n*3+1, len(self.parameter_combos), f'Model applied to {region}', ha='center')

        #add vertical lines to distinguish/delineate the test regions
        for i in range(len(self.app_outnames)):
            ax.axvline(i*3 - 0.5, color='cyan', ls='-', lw=2)

        #annotate the heatmap with performance metric values
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                if self.stats_array[i, j] >= 0.5:
                    color = 'k'
                else:
                    color = 'w'
                _ = ax.text(j, i, self.stats_array[i, j], ha="center", va="center", color=color)

        plt.savefig(self.out_path+'heatmap.png', dpi=400)


    def results_by_training_data(self):
        """
        Creates plots that show performance metrics as a function of training
        data set; one subplot for each of the files to which the model output was applied.
        """

        fig, _ = plt.subplots(1, 3, figsize=(12, 4))

        for (j, _), ax in zip(enumerate(self.nicknames), fig.axes):
            precision = []
            recall =[]
            f_1 = []
            for i in range(self.stats_array.shape[0]):
                precision.append(self.stats_array[i, j*3])
                recall.append(self.stats_array[i, j*3+1])
                f_1.append(self.stats_array[i, j*3+2])
            ax.plot(range(len(precision)), precision, color='b', marker='o',\
                     ls='-.', label='Precision')
            ax.plot(range(len(recall)), recall, color='r', marker='^',\
                     ls=':', label='Recall')
            ax.plot(range(len(f_1)), f_1, color='k', marker='s', ls='-',\
                     label='F1 score')
            if j == 0:
                ax.legend(loc='lower right')

            _ = ax.set_xticks(np.arange(len(self.nicknames)), labels=self.nicknames)
            ax.set_ylabel('Precision/Recall/F1-score')

            ax.set_ylim([0, 1])

            title = self.app_outnames[j]
            title = title.split('_')[-1]
            ax.set_title(f'Model applied to {title}')
        plt.tight_layout()

        plt.savefig(self.out_path+'metrics_by_training_data.png', dpi=400)


if __name__ == "__main__":
    pass
