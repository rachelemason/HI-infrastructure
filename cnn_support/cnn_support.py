#!/usr/bin/env python3
#cnn_support.py
#REM 2022-02-15

"""
Code to support use of the BFGN package (github.com/pgbrodrick/bfg-nets),
which is a package for applying convolutional neural nets to remote
sensing data. Contains classes to help with:
  - setting up and visualizing training data
  - visualizing the applied models
  - looping over different training datasets and
    parameter combinations, and visualizing results
Intended for use with REM's dedicated Jupyter notebooks. Requires bfgn-gpu
environment.
"""

import os
import shutil
import subprocess
import warnings
import glob
import json
import ast
from IPython.utils import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import gdal
import fiona
import rasterio
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.enums import Resampling
from shapely.geometry import shape
import geopandas as gpd
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

GEO_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/geo_data/'

def set_permutations_dict():
    """
    Just a small function to create a dictionary to hold the filenames that will be
    used when we iterate over training datasets. Saves a few lines of code in the
    notebook.
    """

    permutations = {}
    permutations['boundary_files'] = ''
    permutations['feature_files'] = ''
    permutations['response_files'] = ''

    return permutations


class Utils():
    """
    A few tools for file IO, file conversions, getting info from files, etc.
    """

    def __init__(self):
        pass

    @classmethod
    def count_buildings(cls, feature_path, response_path, available_training_sets):
        """
        Count and print the number of polygons (which are generally buildings)
        in each shapefile in available_training_data. Also print the number of
        polygons/buildings in the entire training set.
        """

        building_count = 0
        pixel_count = 0
        for nickname, shpfile in available_training_sets.items():
            if 'HBMain' not in shpfile: #avoid double-counting overlap with HBLower
                geom = fiona.open(response_path+shpfile+'_responses.shp')

                with rasterio.open(feature_path+shpfile+'_hires_surface.tif') as src:
                    pixels = src.shape[0] * src.shape[1]

                print(f"{nickname} contains {pixels} pixels and {len(geom)} features")
                building_count += len(geom)
                pixel_count += pixels
        print(f'Total number of features = {building_count}')
        print(f'Total number of pixels = {pixel_count}')


    @classmethod
    def input_files_from_config(cls, config_file, print_files=True):
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
                        print('here')
                        line = line.replace(kind, '').strip()
                        print('here too', line)
                        files = ast.literal_eval(line)
                        print('and here')
                        filedict[kind] = files
                        if print_files:
                            print(f'{kind} {files}')

        return filedict['feature_files:'], filedict['response_files:'], filedict['boundary_files:']

    @classmethod
    def rasterize(cls, reference_raster, shapefile, tif_out, replace_existing):
        """
        Converts the .shp files in <shapefile_list> into .tif files, using
        the geometry info in <reference_raster>. Output tif files are written
        to the same directory as the input shapefiles.
        """

        feature_set = gdal.Open(reference_raster, gdal.GA_ReadOnly)
        trans = feature_set.GetGeoTransform()

        def do_the_work():
            print(f'Rasterizing {shapefile} using {reference_raster} as reference')
            cmd_str = 'gdal_rasterize ' + shapefile + ' ' + tif_out +\
            ' -init 0 -burn 1 -te ' + str(trans[0]) + ' ' + str(trans[3] +\
            trans[5]*feature_set.RasterYSize) + ' ' + str(trans[0] +\
            trans[1]*feature_set.RasterXSize) + ' ' + str(trans[3]) + ' -tr ' +\
            str(trans[1]) + ' ' + str(trans[5])
            subprocess.call(cmd_str, shell=True)

        if not replace_existing:
            if os.path.isfile(tif_out):
                print(f"{tif_out} already exists, not recreating")
            else:
                do_the_work()
        else:
            do_the_work()


    @classmethod
    def extract_raster_section(cls, main_file, template, destination, replace_existing=True):
        """
        Write the subset of raster <main_file> specified by <template>
        into <destination> tif.
        """

        def do_the_work():
            print(f'Extracting subset of {main_file} defined by {template}')
            geo = gdal.Open(template)
            ulx, xres, _, uly, _, yres = geo.GetGeoTransform()
            lrx = ulx + (geo.RasterXSize * xres)
            lry = uly + (geo.RasterYSize * yres)
            gdal.Translate(destination, gdal.Open(main_file), projWin=[ulx, uly, lrx, lry])

        if not replace_existing:
            if os.path.isfile(destination):
                print(f"{destination} already exists, not recreating")
            else:
                do_the_work()
        else:
            do_the_work()


    def trim_tiles(self, tile_list, in_path, in_suffix, out_path, out_suffix, boundary_file):
        """
        Crop tiles to shapefile boundaries to make at least some of them a more manageable size
        """

        for tile in tile_list:

            print(f'Cropping regions of {tile} that lie outside study region boundaries')
            with fiona.open(boundary_file, "r") as shpf:
                shapes = [feature["geometry"] for feature in shpf]

            with rasterio.open(f'{in_path}{tile}/{tile}{in_suffix}') as f:
                arr, out_trans = mask(f, shapes, crop=True)
                profile = f.profile

            profile.update({"height": arr.shape[1], "width": arr.shape[2], "transform": out_trans,\
                    'compress': 'lzw'})

            #there are 2 tiles that lie in 2 study regions; need to give special names
            if tile in ['tile008', 'tile014'] and 'NKonaSKohala' in boundary_file:
                tile = tile+'a'
            if tile in ['tile008', 'tile014'] and 'NHiloHamakua' in boundary_file:
                tile = tile+'b'

            with rasterio.open(f'{out_path}{tile}{out_suffix}', 'w', **profile) as f:
                f.write(arr)


    @classmethod
    def resample_raster(cls, in_file, out_file, template_file, scale_factor=None,\
                        replace_existing=False):
        """
        Resample a raster, changing the height and width by scale_factor. Written
        for up-sampling NDVI from 2m resolution to 1m pixel size, to match high-
        resolution DSM.
        """

        def do_the_work():
            with rasterio.open(in_file) as src:

                if scale_factor:
                    new_height = int(src.height * scale_factor)
                    new_width = int(src.width * scale_factor)
                else:
                    with rasterio.open(template_file) as template:
                        new_height = template.height
                        new_width = template.width

                # resample data to target shape
                data = src.read(out_shape=(new_height, new_width), resampling=Resampling.bilinear)

                # scale image transform
                transform = src.transform * src.transform.scale((src.width / data.shape[-1]),\
                                                            (src.height / data.shape[-2]))
                #update metadata
                profile = src.profile
                profile.update(transform=transform, driver='GTiff', height=new_height,\
                               width=new_width, crs=src.crs)

            with rasterio.open(out_file, 'w', **profile) as dst:
                dst.write(data)

        if not replace_existing:
            if os.path.isfile(out_file):
                print(f"{out_file} already exists, not recreating")
            else:
                do_the_work()
        else:
            do_the_work()

    @classmethod
    def tif_to_array(cls, tif, get_first_only=False):
        """
        Returns contents of <tif>as a numpy array, optionally containing only the
        first band if it's a multi-band dataset. Some BFGN methods,
        like data_management.apply_model_to_data.apply_model_to_site, write a tif
        instead of returning an array, so this method is useful for further
        operations on those files (but can be used for any tif)
        """

        data = gdal.Open(tif, gdal.GA_ReadOnly)
        arr = data.ReadAsArray()
        if get_first_only and len(arr.shape) > 2:
            arr = arr[0]
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float32) #arrays must be float for next line to work
        arr[arr == data.GetRasterBand(1).GetNoDataValue()] = np.nan

        return arr


    @classmethod
    def boundary_shp_to_mask(cls, boundary_file, background_file):
        """
        Return a numpy array ('masky') with the same shape as <background_file>,
        in which pixels within all polygons in <boundary_file> have values >=0,
        and pixels outside those polygons have value = np.nan.
        """

        boundary_ds = fiona.open(boundary_file, 'r')
        data = gdal.Open(background_file, gdal.GA_ReadOnly)
        geo = data.GetGeoTransform()
        geo = [geo[1], geo[2], geo[0], geo[4], geo[5], geo[3]]
        background = data.ReadAsArray()
        masky = np.full(background.shape, np.nan)
        for _, polygon in enumerate(boundary_ds):
            rasterio.features.rasterize([polygon['geometry']], transform=geo,\
                                        default_value=0, out=masky)
        return masky


    @classmethod
    def create_hillshade(cls, in_file, out_file, replace_existing=False):
        """
        Creates a hillshaded version of in_file and writes it to out_file. Differs
        from self.hillshade in that it takes a file as input (as opposed to array)
        and writes to file instead of displaying a plot.
        """

        def do_the_work():

            print(f'Hillshading {in_file}...')

            with rasterio.open(in_file) as src:
                data = src.read()[0]
                profile = src.profile

            light = LightSource(azdeg=315, altdeg=45)
            hillshade = light.hillshade(data)

            hillshade = np.expand_dims(hillshade, 0).astype(np.float32)

            with rasterio.open(out_file, 'w', **profile) as dst:
                dst.write(hillshade)

        if not replace_existing:
            if os.path.isfile(out_file):
                print(f"{out_file} already exists, not recreating")
            else:
                do_the_work()
        else:
            do_the_work()


    @classmethod
    def hillshade(cls, ax, to_plot):
        """
        Helper function for show_input_data() etc. Plots data on an axis using
        hillshading, using default parameters for sun azimuth and altitude.
        """

        light = LightSource(azdeg=315, altdeg=45)
        ax.imshow(light.hillshade(to_plot), cmap='gray')


    @classmethod
    def create_data_combos(cls, required_data, name, inpath, outpath, location,\
                           replace_existing=False):
        """
        Merge selected bands from different data products (e.g. aMCU and DSM),
        write the resulting array to a tif. Extent and resolution of files must
        match.
        """

        if not replace_existing and os.path.isfile(f'{outpath}{location}_{name}.tif'):
            print(f'{outpath}{location}_{name}.tif exists; not recreating')

        else:
            print(f'Creating {outpath}{location}_{name}.tif')
            pieces = []
            for data_type, bands in required_data.items():
                with rasterio.open(f'{inpath}{location}_{data_type}.tif', 'r') as f:
                    meta = f.meta.copy()
                    arr = f.read()

                    if len(arr.shape) == 2:
                        pieces.append(np.expand_dims(arr, 0))
                    elif len(arr.shape) > 2:
                        for n in bands:
                            pieces.append(np.expand_dims(arr[n], 0))
                    else:
                        print('This is a 1D array, something is wrong')
                        break

            combo = np.vstack(tuple(pieces))

            meta.update({'count': combo.shape[0]})

            with rasterio.open(f'{outpath}{location}_{name}.tif', 'w', **meta) as f:
                f.write(combo)


class TrainingData(Utils):
    """
    Methods for visualizing training data.
    """

    def __init__(self):
        Utils.__init__(self)


    @classmethod
    def create_training_lists(cls, paths, available_training_sets, desired_training_set, lo_res=False):
        """
        Return a properly-formatted list of lists of training data, based on the available
        training data and the desired subset.
        """

        parameter_combos = []
        for train in desired_training_set:
            boundaries = [f"{paths['boundary']}{available_training_sets[item]}_boundary.shp"\
                          for item in train]
            features = []
            responses = []
            for item in train:
                features.append([f"{paths['features']}{available_training_sets[item]}_hires_surface.tif"])
                if not lo_res:
                    responses.append([f"{paths['responses']}{available_training_sets[item]}_responses.tif"])
                else:
                    responses.append([f"{paths['responses']}{available_training_sets[item]}_lores_responses.tif"])
            parameter_combos.append([boundaries, features, responses])

        return parameter_combos


    def show_input_data(self, feature_file, response_file, boundary_file, hillshade=True):
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
            self.hillshade(ax1, map_array)
        else:
            ax1.imshow(map_array)

        # show the boundary of the region(s) containing the training data
        masky = self.boundary_shp_to_mask(boundary_file, feature_file)

        # show where the labelled features are, within the training boundaries
        responses = gdal.Open(response_file, gdal.GA_ReadOnly).ReadAsArray()
        masky[responses != 0] = responses[responses != 0] # put the buildings on the canvas
        ax2.imshow(masky)

        plt.tight_layout()


    @classmethod
    def plot_training_data(cls, features, responses, images_to_plot=3, feature_band=0,\
                           nodata_value=-9999):
        """ Tool to plot the training and response data data side by side. Adapted from
            the ecoCNN code.
            Arguments:
            features - 4d numpy array
            Array of data features, arranged as n,y,x,p, where n is the number of samples,
            y is the data y dimension (2*window_size_radius), x is the data x dimension
            (2*window_size_radius), and p is the number of features.
            responses - 4d numpy array
            Array of of data responses, arranged as n,y,x,p, where n is the number of samples,
            y is the data y dimension (2*window_size_radius), x is the data x dimension
            (2*window_size_radius), and p is the response dimension (always 1).
        """

        features = features.copy()
        responses = responses.copy()
        features[features == nodata_value] = np.nan
        responses[responses == nodata_value] = np.nan

        feat_nan = np.squeeze(np.isnan(features[:, :, :, 0]))
        features[feat_nan, :] = np.nan

        _ = plt.figure(figsize=(4, images_to_plot*2))
        gs1 = gridspec.GridSpec(images_to_plot, 2)
        for n in range(0, images_to_plot):
            _ = plt.subplot(gs1[n, 0])

            feat_min = np.nanmin(features[n, :, :, feature_band])
            feat_max = np.nanmax(features[n, :, :, feature_band])

            plt.imshow(features[n, :, :, feature_band], vmin=feat_min, vmax=feat_max)
            plt.xticks([])
            plt.yticks([])
            if n == 0:
                plt.title('Feature')

            _ = plt.subplot(gs1[n, 1])
            plt.imshow(responses[n, :, :, 0], vmin=0, vmax=1)
            plt.xticks([])
            plt.yticks([])
            if n == 0:
                plt.title('Response')


class AppliedModel():
    """
    Methods for applying models, visualizing applied models and calculating
    performance metrics.
    """

    def __init__(self):
        pass


    def _remove_roads(self, gdf, road_files):
        """
        Helper method for self.(). Reads roads from
        <road_files> shapefiles, applies buffer, removes polygons
        from <gdf> that overlap with buffered roads. Returns edited
        <gdf>.
        """

        for file in road_files:
            roads = gpd.read_file(file)
            roads = roads[roads['island'].str.match('Hawaii')]
            roads.geometry = roads.geometry.buffer(1)
            gdf = gdf.loc[~gdf.intersects(roads.unary_union)].reset_index(drop=True)

        return gdf


    def _remove_coastal_artefacts(self, gdf):
        """
        Helper method for self.clean_coasts_and_roads(). Reads in coastline shapefile
        and removes polygons from <gdf> that intersect with the coast. Same with county
        TML/Parcel map outline. Returns edited <gdf>. Intended for removing coastal
        artefacts.
        """

        #coastline as defined by coastline shapefile
        coast = gpd.read_file(f'{GEO_PATH}Coastline/Coastline.shp')
        coast = coast[coast['isle'] == 'Hawaii'].reset_index(drop=True)
        coast = coast.to_crs(epsg=32605)

        gdf = gdf.loc[gdf.within(coast.unary_union)].reset_index(drop=True)

        #coastline as defined by County TMK map outline
        try:
            parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_outline.shp')
        except fiona.errors.DriverError:
            parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_-_Hawaii_County.shp')
            parcels = parcels.to_crs(epsg=32605)
            parcels['dissolve_by'] = 0
            parcels = parcels.dissolve(by='dissolve_by').reset_index(drop=True)
            parcels = parcels['geometry']
            parcels.to_file(f'{GEO_PATH}Parcels/Parcels_outline.shp', driver='ESRI Shapefile')

        gdf = gdf.loc[gdf.within(parcels.unary_union)].reset_index(drop=True)

        return gdf


    def clean_coasts_and_roads(self, tile, path, road_files):
        """
        Clean up coastal artefacts in an individual tile map by vectorizing the map and
        rejecting polygons that don't lie wholly within (1) the coastline shapefile, and
        (2) the outline of the County TMK/Parcel map. Also, reject polygons that overlap
        with roads (in State DOT shapefiles). Then re-rasterize and save the 'cleaned' file.
        """

        with rasterio.open(f'{path}ndvi_cut_model_{tile}.tif') as f:
            arr = f.read()
            meta = f.meta

        print(f'Vectorizing {path}ndvi_cut_model_{tile}.tif')
        polygons = []
        values = []
        for vec in rasterio.features.shapes(arr.astype(np.int32), transform=meta['transform']):
            polygons.append(shape(vec[0]))
            values.append(vec[1])
        gdf = gpd.GeoDataFrame(crs=meta['crs'], geometry=polygons)

        gdf['Values'] = values

        #Remove polygons that intersect roads (do before getting the bounding rectangle)
        print('  - Removing polygons that intersect with roads')
        gdf = self._remove_roads(gdf, road_files)

        #Remove polygons that are outside the County parcel/TMK map outline, or outside the
        #coastline
        print('  - Removing polygons outside/overlapping the coast')
        gdf = self._remove_coastal_artefacts(gdf)

        #Rasterize and save what should be a 'cleaner' raster than the input one
        print(f'  - Rasterizing and saving {path}cleaner_model_{tile}.tif')
        meta.update({'compress': 'lzw'})
        with rasterio.open(f'{path}cleaner_model_{tile}.tif', 'w+', **meta) as out:
            shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf.Values))
            burned = rasterize(shapes=shapes, fill=0, out_shape=out.shape, transform=out.transform)
            burned[arr[0] < 0] = meta['nodata']
            out.write_band(1, burned)


    def apply_model(self, config_file, application_file, outfile, method='threshold',\
                    ndvi_cut=None, ndvi_file=None):
        """
        Apply an existing model to a new area, convert probabilities to binary classes,
        and write to some sensible storage location. This method is intended to be used to
        apply models to large tiles - this could be done via Loops() but the resulting
        files are so large that they should really be created individually and only when
        really needed
        """

        config = configs.create_config_from_file(config_file)
        data_container = data_core.DataContainer(config)
        data_container.build_or_load_rawfile_data(rebuild=False)
        data_container.build_or_load_scalers()
        data_container.load_sequences()

        experiment = experiments.Experiment(config)
        experiment.build_or_load_model(data_container=data_container)

        print(f'Working on {application_file}')
        apply_model_to_data.apply_model_to_site(experiment.model, data_container,\
                                                [application_file], outfile)

        utils = Utils()
        print(f' -- converting {outfile}.tif to array')
        applied_model = utils.tif_to_array(outfile+'.tif')

        print(' -- getting binary classes')
        classes = self.probabilities_to_classes(method, applied_model,\
                                                tif_template=outfile+'.tif')

        if ndvi_cut and ndvi_file:
            self.apply_ndvi_cut(classes, ndvi_file, ndvi_cut, outfile)


    @classmethod
    def apply_ndvi_cut(cls, classes, ndvi_file, ndvi_cut, outfile=None):
        """
        Apply an NDVI cut to a model array that contains binary classes. Finds the NDVI
        value in <ndvi_file> for each model pixel, and if it is greater than ndvi_cut,
        the class is changed from 1 to 0. This was written to exclude trees
        and the few water pixels that were incorrectly identified as buildings.
        """

        print(' -- applying NDVI cut')
        print(ndvi_file)
        with rasterio.open(ndvi_file, 'r') as f:
            meta = f.meta.copy()
            ndvi = f.read()
        cut_classes = np.expand_dims(classes, 0).astype(np.float32)
        cut_classes[ndvi > ndvi_cut] = 0
        if outfile is not None:
            with rasterio.open(f"{outfile.replace('applied', 'ndvi_cut')}.tif", 'w', **meta) as f:
                f.write(cut_classes)

        return cut_classes


    def probabilities_to_classes(self, method, applied_model_arr, threshold_val=0.9,\
                                 nodata_value=-9999, tif_template=None):
        """
        Converts the class probabilities in the applied_model array (not tif) into
        binary classes using maximum likelihood or a threshold.
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

        if tif_template:
            self.binary_classes_to_tif(output, tif_template, method)

        return output


    @classmethod
    def binary_classes_to_tif(cls, arr, template, method):
        """
        Create tif containing the model created by self.probabilities_to_classes()
        """

        with rasterio.open(template, 'r') as f:
            meta = f.meta.copy()
        arr = np.expand_dims(arr, 0).astype(np.float32)
        meta.update({'count': 1})
        name = template.replace('applied', method)
        with rasterio.open(name, 'w', **meta) as f:
            f.write(arr)


    @classmethod
    def show_applied_model(cls, applied_model, original_img, zoom, ax2_data, responses=None,\
                           hillshade=True, filename=None):
        """
        Plots the applied model created by
        bfgn.data_management.apply_model_to_data.apply_model_to_site, converted to a numpy array by
        self.applied_model_as_array. Also zooms into a subregion and shows (1) the applied model
        probabilities, (2) the applied model converted to classes, and (3) the original image.
        """

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

        if responses is not None:
            shp = gdal.Open(responses, gdal.GA_ReadOnly)
            shp = shp.ReadAsArray()
            shp = np.ma.masked_where(shp < 1.0, shp)

        #Show the applied model probabilities for class 2 over the whole area to which it
        #has been applied
        cbar = ax1.imshow(applied_model[1], cmap='GnBu_r')
        plt.colorbar(cbar, ax=ax1, shrink=0.6, pad=0.01)
        if responses is not None:
            ax1.imshow(shp, alpha=0.7, cmap='Reds_r')

        #Convert probability map into binary classes using threshold
        ax2.imshow(ax2_data, cmap='Greys')
        if responses is not None:
            ax2.imshow(shp, alpha=0.7, cmap='Reds_r')
        #Show the area to be zoomed into
        ax2.add_patch(Rectangle((zoom[2], zoom[0]), zoom[3]-zoom[2], zoom[1]-zoom[0],\
                               fill=False, edgecolor='r'))

        #Zoom into the specified subregion
        ax3.imshow(ax2_data[zoom[0]:zoom[1], zoom[2]:zoom[3]], cmap='Greys')
        #Overplot responses (buildings), if available
        if responses is not None:
            ax3.imshow(shp[zoom[0]:zoom[1], zoom[2]:zoom[3]], alpha=0.8, cmap='Reds_r')

        #The original image for which everything was predicted (same subset/zoom region;
        #only show the first band if there are >1)
        utils = Utils()
        original = utils.tif_to_array(original_img, get_first_only=True)
        original = original[zoom[0]:zoom[1], zoom[2]:zoom[3]]

        if hillshade:
            utils.hillshade(ax4, original)
        else:
            ax4.imshow(original)

        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, dpi=400)


    @classmethod
    def print_stats(cls, statsdict):
        """
        Pretty print the dictionary of performance stats returned by
        self.performance_metrics.  Using simple string formatting as
        I can't install pandas in this environment.
        """

        print('Class       | Precision | Recall | F1-score | Support\n')
        for key, vals in statsdict.items():
            try:
                vals = [np.round(v, 2) for v in vals.values()]
                spc = ' ' * (18-len(key))
                print(f'{key}{spc}{vals[0]}      {vals[1]}      {vals[2]}      {vals[3]}')
            except AttributeError:
                print(f'{key}    {np.round(vals, 2)}')


    @classmethod
    def performance_metrics(cls, classes, responses, boundary_file):
        """
        Given a model class prediction array and a set of responses, calculate precision,
        recall, and f1-score for each class (currently assumes binary classes)
        """

        # create an array of the same shape as the applied model 'canvas'
        # in which everything outside the training dataset boundary/boundaries is NaN
        utils = Utils()
        masky = utils.boundary_shp_to_mask(boundary_file, responses)
        response_array = utils.tif_to_array(responses)

        # insert the labelled responses into the array, inside the training boundaries
        masky[response_array != 0] = response_array[response_array != 0]

        # flatten to 1D and remove NaNs
        predicted = classes.flatten()
        expected = masky.flatten()
        predicted = list(predicted[~(np.isnan(expected))])
        expected = list(expected[~(np.isnan(expected))])

        # get performance metrics
        stats = classification_report(expected, predicted, output_dict=True)

        return stats


class Loops(Utils, AppliedModel):
    """
    Methods for looping over BFGN parameter combinations and evaluating results.
    """

   #using settatr in __init__ is nice but causes pylint to barf, so
   #pylint: disable=no-member

    def __init__(self, application_data, iteration_data, settings_file):
        self.data_types = [] #optionally filled in in iteration_data
        for key in application_data:
            setattr(self, key, application_data[key])
        for key in iteration_data:
            setattr(self, key, iteration_data[key])
        self.settings_file = settings_file
        self.ml_stats = None #defined in _create_stats_array()
        self.threshold_stats = None #defined in _create_stats_array()
        self.cut_stats = None #defined in _create_stats_array()

        Utils.__init__(self)
        AppliedModel.__init__(self)


    @classmethod
    def _train_or_load_model(cls, idx, rebuild, fit_model, outdir):
        """
        Helper method for self.loop_over_configs. Train a CNN or load an existing model.
        Returns the BFGN Experiment and DataContainer objects that are needed for applying
        models to data.
        """

        config = configs.create_config_from_file(f'new_settings_{idx}.yaml')

        if rebuild:
            for f in glob.glob(outdir+'munged*'):
                os.remove(f)

        #Remove any existing model files; doing this outside lower if fit_model
        #loop so that model.h5 file is definitely gone before starting training
        if fit_model:
            for pattern in ['app*', 'config*', 'log*', 'model*', 'data_container*',\
                            'stats*']:
                for f in glob.glob(outdir+pattern):
                    os.remove(f)
            shutil.rmtree(outdir+'tensorboard', ignore_errors=True)

        data_container = data_core.DataContainer(config)
        data_container.build_or_load_rawfile_data(rebuild=rebuild)
        data_container.build_or_load_scalers()
        data_container.load_sequences()
        n_samples = np.sum([len(n) for n in data_container.features])
        print(f'Training dataset contained {n_samples} samples')

        experiment = experiments.Experiment(config)
        experiment.build_or_load_model(data_container=data_container)

        if fit_model:
            print('Fitting model and creating report...')
            with io.capture_output():
                experiment.fit_model_with_data_container(data_container, resume_training=False)
                final_report = bfgn.reporting.reports.Reporter(data_container, experiment,\
                                                                   config)
                final_report.create_model_report()
                plt.close('all')

        return experiment, data_container


    def _apply_model(self, idx, outdir, threshold=0.90, ndvi_cut=None):
        """
        Helper method for self.loop_over_configs. Apply an existing trained model
        to data (test fields), write image files, and return performance metrics.
        """

        #retrieve the fitted model
        experiment, data_container = self._train_or_load_model(idx, rebuild=False,\
                                                              fit_model=False, outdir=outdir)

        #for each test region, apply the model and get stats
        ml_stats = []
        threshold_stats = []
        cut_stats = []
        with io.capture_output():
            for i, _f in enumerate(self.app_features):

                #find the type of input data used and update application filename to match
                #BFGN needs application data with same number of bands as model training data
                if self.data_types:
                    if self.data_types[idx] in self.parameter_combos[idx][1][0][0]:
                        data_type = self.data_types[idx]
                        _f[0] = _f[0].replace('hires_surface', self.data_types[idx])
                        print('Changing filename', _f[0])
                else:
                    data_type = 'hires_surface'

                apply_model_to_data.apply_model_to_site(experiment.model, data_container,\
                                                            _f, outdir+self.app_outnames[i])

                applied_model = self.tif_to_array(outdir+self.app_outnames[i]+'.tif')

                #convert probabilities to binary classes by both thresholding and
                #maximum likeihood
                ml_classes = self.probabilities_to_classes('ML', applied_model,\
                                                                  threshold_val=threshold)
                threshold_classes = self.probabilities_to_classes('threshold', applied_model,\
                                                                  threshold_val=threshold)

                #apply NDVI threshold to thresholded classes
                if any(s in data_type for s in ['hires_surface', 'hillshade', 'eigen']):
                    ndvi_file = _f[0].replace(data_type, 'ndvi_hires')
                else:
                    ndvi_file = _f[0].replace(data_type, 'ndvi')
                cut_classes = self.apply_ndvi_cut(threshold_classes, ndvi_file, ndvi_cut,\
                                                        outfile=outdir+self.app_outnames[i])

                #make pdf showing applied model
                hillshade = bool('res_surface' in _f[0] or 'hillshade' in _f[0])
                self.show_applied_model(applied_model=applied_model, zoom=self.zooms[i],\
                                                original_img=_f[0], hillshade=hillshade,\
                                                responses=self.app_responses[i],\
                                                ax2_data=cut_classes[0],\
                                                filename=f"{outdir}{self.app_outnames[i]}.pdf")

                #record stats for ML thresholded and NDVI-cut models
                ml_stats.append(self.performance_metrics(ml_classes, self.app_responses[i],\
                                                self.app_boundaries[i]))
                threshold_stats.append(self.performance_metrics(threshold_classes,\
                                                                self.app_responses[i],\
                                                                self.app_boundaries[i]))
                cut_stats.append(self.performance_metrics(cut_classes, self.app_responses[i],\
                                                self.app_boundaries[i]))

                if self.data_types:
                    _f[0] = _f[0].replace(self.data_types[idx], 'hires_surface')

        #store the performance stats (if any) for later use
        for arr, name in zip([ml_stats, threshold_stats, cut_stats],\
                             ['stats_ML', 'stats_threshold', 'stats_cut']):
            with open(f'{outdir}{name}.json', 'w', encoding='utf-8') as f:
                json.dump(arr, f)

        return ml_stats, threshold_stats, cut_stats


    def _create_stats_arrays(self, stats):
        """
        Helper method for self.loop_over_configs. Convert big unwieldy list of
        performance metrics created by self.loop_over_configs to self.stats_array
        which is understood by plotting methods.
        """

        #Array to hold the performance metrics
        stats_array = np.zeros((len(self.parameter_combos),\
                                     len(self.app_features)*3)) #3 is for the 3 metrics

        for i, _ in enumerate(stats):
            for idx, data in enumerate(stats[i]):
                #stats for class '1.0' (buildings)
                stats_array[i, idx*3] = np.round(data['1.0']['precision'], 2)
                stats_array[i, idx*3+1] = np.round(data['1.0']['recall'], 2)
                stats_array[i, idx*3+2] = np.round(data['1.0']['f1-score'], 2)

        return stats_array


    def loop_over_configs(self, rebuild_data=False, fit_model=True, apply_model=True,\
                          use_existing=True, threshold=0.95, ndvi_cut=None):
        """
        Loops over BFGN configurations (can be training data or other parameters in the
        config file), and returns a numpy array of model performance metrics (precision,
        recall, F1-score) for all combinations.
        This function intentionally produces much less diagnostic info than if the model
        were fit outside it, using direct calls to BFGN; looping over parameters assumes
        the user already knows what they're doing.
        """

        #Loop over the parameter combos, fit and/or apply model
        ml_stats = []
        threshold_stats = []
        cut_stats = []
        for idx, params in enumerate(self.parameter_combos):
            #get the current set of parameters
            combo_dict = {}
            for i, j in enumerate(params):
                name = list(self.permutations.keys())[i]
                combo_dict[name] = j

            print('\n===================================================')
            print(f'Working on parameter combination #{idx}:\n')
            #for key, value in combo_dict.items():
            #    print(f'{key}: {value}')
            #create a new settings file with these parameters
            self._create_settings_file(combo_dict, idx)

            outdir = f'{self.out_path}combo_{idx}/'
            os.makedirs(outdir, exist_ok=True)

            #fit model (or not, if existing results OK). Models are written to
            #.h5 files so nothing is returned here.
            if fit_model:
                if use_existing:
                    if os.path.exists(f'{outdir}model.h5'):
                        print(f'***Model {outdir}model.h5 exists; nothing to do here')
                    else:
                        print(f'***No model found for {outdir}model.h5; will train a new one')
                        _, _ = self._train_or_load_model(idx, rebuild_data, fit_model,\
                                                         outdir)
                else:
                    _, _ = self._train_or_load_model(idx, rebuild_data, fit_model,\
                                                     outdir)

            #apply model to data or retrieve existing performance stats
            if apply_model:
                if use_existing:
                    try:
                        with open(f'{outdir}stats_ML.json', 'r', encoding='utf-8') as f:
                            stats0 = json.load(f)
                        with open(f'{outdir}stats_threshold.json', 'r', encoding='utf-8') as f:
                            stats1 = json.load(f)
                        with open(f'{outdir}stats_cut.json', 'r', encoding='utf-8') as f:
                            stats2 = json.load(f)
                        print(f'Loaded stats for {outdir} from file')
                    except FileNotFoundError:
                        print(f'Saved stats not found in {outdir}; applying model')
                        stats0, stats1, stats2 = self._apply_model(idx, outdir, threshold, ndvi_cut)
                        plt.close('all')
                else:
                    stats0, stats1, stats2 = self._apply_model(idx, outdir, threshold, ndvi_cut)
                    plt.close('all')
                ml_stats.append(stats0)
                threshold_stats.append(stats1)
                cut_stats.append(stats2)

            #Archive the settings file for this parameter combo
            shutil.move(f'new_settings_{idx}.yaml', f'{outdir}new_settings_{idx}.yaml')

        #Once all stats have been gathered, reformat nicely
        if apply_model:
            self.ml_stats = self._create_stats_arrays(ml_stats)
            self.threshold_stats = self._create_stats_arrays(threshold_stats)
            self.cut_stats = self._create_stats_arrays(cut_stats)


    def _create_settings_file(self, combo_dict, num):
        """
        Helper method for self.loop_over_configs. Writes the settings file
        needed for the model to run with new parameter or input file settings.
        Returns: nothing.
        """

        newlines = []
        with open(self.settings_file, 'r', encoding='utf-8') as f:
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
                #make separate model/data output directories for different parameter combos -
                elif 'dir_out:' in line:
                    line = f'  dir_out: {self.out_path}combo_{num}/\n'
                    newlines.append(line)
                else:
                    newlines.append(line)

        #create a new settings file and output dir for this parameter combo
        with open(f'new_settings_{num}.yaml', 'w', encoding='utf-8') as f:
            for line in newlines:
                f.write(line)


    def parameter_heatmap(self, stats_type):
        """
        Plots a heatmap of precision, recall and F1 score for a set of model test regions and
        parameter combinations, using <stats_array> produced by the loop_over_configs function.
        """

        print(f'Parameters tested:{list(self.permutations.keys())}')

        to_plot = self._check_stats_type(stats_type)

        _, ax = plt.subplots(figsize=(20, 20))
        img = ax.imshow(to_plot, vmin=0, vmax=1, cmap='hot')
        _ = plt.colorbar(img, shrink=0.765, pad=0.01, aspect=40)

        #add labels to the plot
        #the basic x labels - metrics used
        xlabels = [['precision', 'recall', 'f1-score'] * len(self.app_outnames)][0]

        #ylabels - parameter combos, or nicknames for them
        if self.nicknames is not None:
            ylabels = self.nicknames
        else:
            ylabels = self.parameter_combos

        _ = ax.set_xticks(np.arange(len(xlabels)), labels=xlabels, rotation=45)
        _ = ax.set_yticks(np.arange(len(ylabels)), labels=ylabels)

        #add labels for the test regions the stats refer to
        test_regions = [a.split('_model_')[1] for a in self.app_outnames]
        for n, region in enumerate(test_regions):
            ax.text(n*3+1, -1, f'Model applied to\n {region}', ha='center')

        #add vertical lines to distinguish/delineate the test regions
        for i in range(len(self.app_outnames)):
            ax.axvline(i*3 - 0.5, color='cyan', ls='-', lw=2)

        #annotate the heatmap with performance metric values
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                if to_plot[i, j] >= 0.5:
                    color = 'k'
                else:
                    color = 'w'
                _ = ax.text(j, i, to_plot[i, j], ha="center", va="center", color=color)

        plt.savefig(self.out_path+'heatmap.png', dpi=400)


    def _check_stats_type(self, stats_type):
        """
        Helper method for plotting functions, returns correct array to plot
        based on stats_type parameter
        """

        if stats_type == 'ML':
            to_plot = self.ml_stats
        elif stats_type == 'threshold':
            to_plot = self.threshold_stats
        elif stats_type == 'cut':
            to_plot = self.cut_stats
        else:
            raise ValueError('Value of stats_type must be one of ML|threshold|cut')

        return to_plot


    def results_by_training_data(self, stats_type):
        """
        Creates plots that show performance metrics as a function of training
        data set; one subplot for each of the files to which the model output was applied.
        """

        to_plot = self._check_stats_type(stats_type)

        fig, _ = plt.subplots(3, 4, figsize=(16, 12))

        for (j, _), ax in zip(enumerate(self.app_outnames), fig.axes):
            if self.app_types[j] == 'neighbour':
                colours = ['lightblue', 'blue', 'black']
            else:
                colours = ['blueviolet', 'magenta', 'lightpink']

            precision = []
            recall = []
            f_1 = []
            for i in range(to_plot.shape[0]):
                precision.append(to_plot[i, j*3])
                recall.append(to_plot[i, j*3+1])
                f_1.append(to_plot[i, j*3+2])
            ax.plot(range(len(recall)), recall, color=colours[1], marker='^',\
                        ls=':', label='Recall')
            ax.plot(range(len(f_1)), f_1, color=colours[2], marker='s', ls='-',\
                        label='F1 score')
            ax.plot(range(len(precision)), precision, color=colours[0], marker='o',\
                        ls='-.', label='Precision')
            if 'HOVE1' in self.nicknames:
                ax.axvline(x=3.5, color='0.5', ls='--')
            if j == 0:
                ax.legend(loc='lower right')
                ax.set_ylabel('Recall/Precision/F1-score')

            _ = ax.set_xticks(np.arange(len(self.nicknames)), labels=self.nicknames,\
                             rotation=45, ha='right')
            ax.set_xlabel('Model training dataset')

            ax.set_ylim([0, 1])

            title = self.app_outnames[j]
            title = title.split('_')[-1]
            ax.set_title(f'Model applied to {title}')

        for idx, ax in enumerate(fig.axes):
            if idx >= len(self.app_outnames):
                ax.axis('off')

        plt.tight_layout()

        plt.savefig(f'{self.out_path}metrics_by_training_data_{stats_type}.png', dpi=400)


    def results_by_test_area(self, training_set_index=8, stats_type='cut'):
        """
        Create a plot that shows performance stats of a single model, fit to several labeled
        test regions.
        """

        to_plot = self._check_stats_type(stats_type)

        _, ax = plt.subplots(1, 1, figsize=(8, 6))

        #get the data out of the relevant stats array
        metrics = {'recall': [], 'precision': [], 'f1-score': []}
        colors = {'recall': [], 'precision': [], 'f1-score': []}
        symbols = {'recall': '^', 'precision': 'o', 'f1-score': 's'}
        for (j, _) in enumerate(self.app_outnames):
            if self.app_types[j] == 'neighbour':
                colors['recall'].append('blue')
                colors['precision'].append('lightblue')
                colors['f1-score'].append('black')
            else:
                colors['recall'].append('blueviolet')
                colors['precision'].append('magenta')
                colors['f1-score'].append('lightpink')
            metrics['precision'].append(to_plot[training_set_index, j*3])
            metrics['recall'].append(to_plot[training_set_index, j*3+1])
            metrics['f1-score'].append(to_plot[training_set_index, j*3+2])

        #set up for a legend that includes mean values of metrics
        labels = {'recall': '', 'precision': '', 'f1-score': ''}
        for metric, values in metrics.items():
            labels[metric] = (f'{metric} (mean={np.round(np.mean(values), 2)})')

        #plot the data + legend
        for metric, values in metrics.items():
            ax.scatter(range(len(values)), values, color=colors[metric],\
                       marker=symbols[metric], s=100, label=labels[metric])
        ax.legend(loc='lower right', fontsize=12)

        #axis stuff
        ax.set_ylabel('Recall/Precision/F1-score', fontsize=14)
        labels = [s.split('applied_model_')[1] for s in self.app_outnames]
        _ = ax.set_xticks(np.arange(len(self.app_outnames)), labels=labels,\
                             rotation=45, ha='right', fontsize=12)
        ax.set_xlabel('Model test region', fontsize=14)
        ax.set_ylim([0, 1])

        plt.savefig(f'{self.out_path}metrics_by_test_area_{stats_type}.png', dpi=400)


if __name__ == "__main__":
    pass
