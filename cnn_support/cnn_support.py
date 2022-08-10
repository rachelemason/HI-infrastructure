#!/usr/bin/env python3
#cnn_support.py
#REM 2022-08-10

"""
Contains functions to help with setting up and visualizing training data with the BFGN package
"""

import os
import subprocess
import ast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import gridspec
import gdal
import fiona
import rasterio
from sklearn.metrics import classification_report


def input_files_from_config(config_file, print_files=True):
    """
    Return a dictionary containing the lists of feature, response, and boundary
    files specified in <config_file>. Dictionary keys are 'feature_files',
    'response_files', and 'boundary_files'
    """

    filedict = {}
    with open(config_file, 'r') as f:
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


def show_applied_model(applied_model, original_img, zoom, responses=None, hillshade=True):
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


if __name__ == "__main__":
    pass
