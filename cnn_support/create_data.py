#!/usr/bin/env python3
#create_data.py
#REM 2022-04-18


"""
Code for creating datasets to be used for training and applying models
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import fiona
import matplotlib.pyplot as plt

GAO_PATH = '/data/gdcsdata/HawaiiMapping/Full_Backfilled_Tiles/'
FEATURE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'
SHAPEFILE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/study_region_boundaries/'

def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def crop_tiles(tile, band, data_suffix, shapefile, outname, crop_nans):
    """
    
    """

    if data_suffix != 'hillshade.tif':
        infile = f'{GAO_PATH}{tile}/{tile}_{data_suffix}'
    else:
        #I derived these files from GAO products so they are in a different location
        infile = f'{FEATURE_PATH}/{tile}_{data_suffix}'

    print(' -- cropping to study region boundaries')
    with fiona.open(f'{SHAPEFILE_PATH}{shapefile}', "r") as shpf:
        shapes = [feature["geometry"] for feature in shpf]

    with rasterio.open(infile) as f:
        cropped, out_trans = mask(f, shapes, crop=True)
        meta = f.meta

    #work on only the band in question, but preserve the number of dimensions
    cropped = cropped[band]
    cropped = np.expand_dims(cropped, 0)
    cropped[cropped == meta['nodata']] = np.nan
    
    #the hillshade tiles, weirdly, have fixed numerical values where they should be NaN
    #use the DSM tiles to fix that (need to save as templates before cropping any
    #null rows/columns - should really just save mask or indices, but it's too late now)
    if data_suffix == 'backfilled_surface_1mres.tif':
        meta.update({"height": cropped.shape[1], "width": cropped.shape[2], "transform": out_trans,\
                'compress': 'lzw', 'nodata': np.nan, 'crs': meta['crs']})
        with rasterio.open(f'{FEATURE_PATH}{tile}_DSM_template.tif', "w", **meta) as f:
                f.write(cropped)
    if data_suffix == 'hillshade.tif':
        dsm_name = f'{FEATURE_PATH}{tile}_DSM_template.tif'
        try:
            with rasterio.open(dsm_name, 'r') as f:
                dsm = f.read()
            cropped = np.where(np.isnan(dsm[0]), np.nan, cropped)
        except rasterio.RasterioIOError:
            raise IOError(f"Cannot use {dsm_name} to fix hillshade; file does not exist")

    if crop_nans:
        #only do this if it would mean cropping same pieces of 1m and 2m data?
        print(' -- removing rows and columns that contain only null values (if any)')
        original_shape = cropped.shape
        nan_cols = np.all(np.isnan(cropped), axis=(0, 1))
        nan_rows = np.all(np.isnan(cropped), axis=(0, 2))
        cropped = cropped[:, ~nan_rows, :]
        cropped = cropped[:, :, ~nan_cols]
        
        #recreate the georeferencing info
        #I think a+e are resolution, b+d are rotation (not relevant), and xoff and yoff define
        #the lat and long of the top-left corner
        a, b, xoff, d, e, yoff, *_ = out_trans
    
        #find the new origin in x, but only if we've clipped columns from the start
        #(left) of the raster
        if nan_cols[0]: #if first column was NaN
            xoff = xoff + (original_shape[2]-cropped.shape[2]) * a
        
        #same for y
        if nan_rows[0]: #if first row was NaN
            yoff = yoff + (original_shape[1]-cropped.shape[1]) * e

        out_trans = rasterio.Affine(a, b, xoff, d, e, yoff)
        new_height, new_width = cropped.shape[1:]
        
    else:
        new_height, new_width = cropped.shape[1:]

    plt.imshow(cropped[0])
    plt.show()
    
    meta.update({"height": cropped.shape[1], "width": cropped.shape[2], "count":cropped.shape[0],\
                 "transform": out_trans, 'compress': 'lzw', 'nodata': np.nan, 'crs': meta['crs']})
    print(f' -- writing {FEATURE_PATH}{tile}_{outname}.tif')
    with rasterio.open(f'{FEATURE_PATH}{tile}_{outname}.tif', "w", **meta) as f:
        f.write(cropped)
    
    file_info = os.stat(f'{FEATURE_PATH}{tile}_{outname}.tif')
    print(convert_bytes(file_info.st_size))


def _create_mosaic(tiles, data_suffix, shapefile):
    """
    Mosaic the tmp files created by _crop_nan_regions; save the mosaic as
    another tmp file
    """
    
    print(f'Mosaicking tiles')
    to_mosaic = []
    for tile in tiles:
        raster = rasterio.open(f'{FEATURE_PATH}tmp{tile}.tif')
        to_mosaic.append(raster)
        meta = raster.meta

    mosaic, out_trans = merge(to_mosaic)

