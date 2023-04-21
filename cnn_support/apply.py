#!/usr/bin/env python3
#apply.py
#REM 2022-04-20


"""
Code for applying model to whole tiles
"""

import os
import pickle
import numpy as np
from scipy import linalg
import rasterio
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.mask import mask
import fiona
from performance import Utils, MapManips

GAO_PATH = '/data/gdcsdata/HawaiiMapping/Full_Backfilled_Tiles/'
FEATURE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'
MODEL_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/bfgn_output_buildings2/model_runs/'
SHAPEFILE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/study_region_boundaries/'


def cnn_average_interpolate(tile, combos):
    """
    Create an average CNN-based map at native (1,) pixel size, then interpolate
    to 2m pixel size
    """

    utils = Utils(None)

    cnns = []
    for combo in combos:
        fname = f'{MODEL_PATH}combo_{combo}/applied_model_tile{tile}.tif'
        # look for a gzipped file first, if it's not there then just use plain tif
        try:
            with rasterio.open(fname+'gz') as src:
                cnn = src.read()
                meta = src.meta
        except rasterio.RasterioIOError:
            with rasterio.open(fname) as src:
                cnn = src.read()
                meta = src.meta
        cnns.append(cnn[0:1, :, :])

    print('Creating mean map')
    meanmap = np.mean(cnns, axis=0)

    meta.update({'count': meanmap.shape[0], 'compress':'lzw'})
    to_file = f'{MODEL_PATH}ensembles/mean_probability_tile{tile}.tif'
    print(f'Writing {to_file}')
    with rasterio.open(to_file, 'w', **meta) as f:
        f.write(meanmap)

    #now interpolate
    out_file = f'{MODEL_PATH}ensembles/tile{tile}_lores_model.tif'
    template = f'{GAO_PATH}tile{tile}/tile{tile}_backfilled_canopyheight_avg.tif'
    utils.resample_raster(to_file, out_file, template)


def loop_over_windows(tile, bad_bands, model, bnorm):
    """
    Apply a saved XGB model to a tile, window by window. Then write out the
    applied model for the whole tile.
    """

    # define refl bands that will be included
    good_bands = [b for b in range(214) if b not in bad_bands]

    # get the XGB saved model
    with open(f'{MODEL_PATH}ensembles/{model}', "rb") as f:
        xgb_mod = pickle.load(f)

    applied_model_windows = []

    for n in range(25):
        window = Window.from_slices((n*500, (n*500)+500), (0, 12500))
        print(n, window)

        # get the refl data
        with rasterio.open(f'{GAO_PATH}tile{tile}/tile{tile}_mosaic_refl') as src:
            refl = src.read(good_bands, window=window).astype(np.float32)
            meta = src.meta

        # brightness-normalize, if needed (do before setting NaNs)
        if bnorm:
            refl = refl / linalg.norm(refl, axis=0)

        # change -9999.0 to NaN so these end up as NaN in final applied models
        # must do after brightness normalization
        refl[refl==meta['nodata']] = np.nan

        # get the tch map, set regions with no refl data to NaN
        with rasterio.open(f'{GAO_PATH}tile{tile}/tile{tile}_backfilled_canopyheight_avg.tif') as src:
            tch = src.read(window=window)
        tch = np.where(np.isnan(refl[0]), np.nan, tch)

        # get the CNN average map at 2m pixel size
        with rasterio.open(f'{MODEL_PATH}ensembles/tile{tile}_lores_model.tif') as src:
            cnn = src.read(window=window)
        cnn = np.where(np.isnan(refl[0]), np.nan, cnn)

        # combine refl, tch, amd meanmap into array with right shape for model
        xdata = np.vstack((refl, cnn, tch))
        original_shape = xdata[0].shape
        xdata = np.reshape(xdata, (xdata.shape[0], -1)).T

        # apply model
        predicted = xgb_mod.predict_proba(xdata)
        predicted = predicted[:, 1:2]
        predicted = predicted.T.reshape(original_shape)

        # try to make sure we have NaNs in the right place (why aren't they already there?)
        # this only works when bnorm=False but that's useful later
        predicted = np.where(np.isnan(refl[0]), np.nan, predicted)

        applied_model_windows.append(predicted)

    applied_model_tile = np.vstack(applied_model_windows)
    applied_model_tile = np.expand_dims(applied_model_tile, axis=0)

    meta.update({'count': applied_model_tile.shape[0], 'nodata': np.nan, 'dtype': np.float32})
    run = model.replace('gb_', '').replace('.pkl', '')
    print(f'Writing {MODEL_PATH}ensembles/{run}_applied_{tile}.tif')
    with rasterio.open(f'{MODEL_PATH}ensembles/{run}_applied_{tile}.tif', 'w', **meta) as f:
        f.write(applied_model_tile)


def xgb_ensemble_classes(tile, threshold=0.2):
    """
    Average the four applied models and convert from probabilities to classes.
    Set 4 rows/columns at each edge of tile to NaN, as they are full of false positives.
    This is probably to do with the window sizes using in the CNN model and may have been
    avoidable by applying CNNs to mosaics instead of full tiles. However, I found that
    that wasn't practical.
    """

    to_average = []
    for run in [1, 2, 3, 4]:
        fname = f'{MODEL_PATH}ensembles/run{run}_applied_{tile}.tif'
        with rasterio.open(fname) as src:
            run_data = src.read()
            meta = src.meta
        to_average.append(run_data)

    print('Creating mean XGB map')
    meanmap = np.mean(to_average, axis=0)

    classmap = np.where(meanmap >= threshold, 1, 0)

    #Make sure there no-data regions are NaN, using a not-brightness-normalized run
    #as the template
    classmap = np.where(np.isnan(to_average[2]), np.nan, classmap)

    #Set dodgy tile edges to NaN
    #print('Setting edge pixels to NaN; means there will be an 8-pixel gap bwteeen tiles')
    classmap[0, :4, :] = np.nan
    classmap[0, -4:, :] = np.nan
    classmap[0, :, :4] = np.nan
    classmap[0, :, -4:] = np.nan

    to_file = f'{MODEL_PATH}ensembles/xgb_class_tile{tile}.tif'
    print(f'Writing {to_file}')
    with rasterio.open(to_file, 'w', **meta) as f:
        f.write(classmap)


def mosaic_and_crop(tiles, boundary_file, region):
    """
    Mosaic the building class tiles and crop to study region boundaries
    """

    to_mosaic = []
    for tile in tiles:
        raster = rasterio.open(f'{MODEL_PATH}ensembles/xgb_class_tile{tile}.tif', 'r')
        to_mosaic.append(raster)
        meta = raster.meta

    #create mosaic and write to file needed for cropping step
    mosaic, out_trans = merge(to_mosaic)
    meta.update({'transform': out_trans, 'height': mosaic.shape[1], 'width': mosaic.shape[2]})
    with rasterio.open(f'{MODEL_PATH}ensembles/tmp.tif', 'w', **meta) as f:
        f.write(mosaic)
    
    #get shapefile and crop mosaic to boundaries
    print(' -- cropping to study region boundaries')
    with fiona.open(f'{SHAPEFILE_PATH}{boundary_file}', "r") as shpf:
        shapes = [feature["geometry"] for feature in shpf]

    with rasterio.open(f'{MODEL_PATH}ensembles/tmp.tif') as f:
        cropped, out_trans = mask(f, shapes, crop=True)
        meta = f.meta
        
    meta.update({'transform': out_trans, 'height': cropped.shape[1], 'width': cropped.shape[2]})
    print(f'Saving {MODEL_PATH}ensembles/{region}_xgb_mosaic.tif')
    with rasterio.open(f'{MODEL_PATH}ensembles/{region}_xgb_mosaic.tif', 'w', **meta) as f:
        f.write(cropped)

    os.remove(f'{MODEL_PATH}ensembles/tmp.tif')
