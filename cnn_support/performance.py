#!/usr/bin/env python3
#performance.py
#REM 2022-04-16


"""
Code for deriving mapping products from probability maps produced by
run_models.py/RunModels.ipynb, and creating various diagnostic plots.
"""

import os
import glob
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import fiona
from shapely.geometry import shape, MultiPolygon, Polygon, box
from shapely.ops import nearest_points
from osgeo import gdal
import geopandas as gpd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skl_PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,\
jaccard_score, r2_score
import shap
from scipy.stats import randint, hmean, pearsonr
from scipy import ndimage, linalg
import seaborn as sns

pd.set_option('display.precision', 2)

RANDOM_STATE = 42
FEATURE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'
RESPONSE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_buildings2/'
BOUNDARY_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_boundaries/'
GEO_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/geo_data/'
ROAD_FILES = [f'{GEO_PATH}SDOT_roads/{f}' for f in ['state_routes_SDOT_epsg32605.shp',\
                                                    'county_routes_SDOT_epsg32605.shp',\
                                                    'ServiceAndOtherRoads_SDOT_epsg32605.shp']]
FIGURE_OUTPUT_PATH = '/home/remason2/figures/'


class Utils():
    """
    Generic helper methods used by >1 class
    """

    def __init__(self, all_labelled_data):
        self.all_labelled_data = all_labelled_data


    def _get_map_and_ref_data(self, map_file, region, resolution=''):
        """
        Helper method that just reads the map and labelled response
        data for the specified region, clips the edges, and returns
        both as 2D arrays
        """

        #open the map
        with rasterio.open(map_file, 'r') as f:
            map_arr = f.read()

        #open the corresponding reference (labelled buildings) file
        ref_file = f'{RESPONSE_PATH}{self.all_labelled_data[region]}_{resolution}responses.tif'
        with rasterio.open(ref_file, 'r') as f:
            ref_arr = f.read()

        map_arr = map_arr[0]
        ref_arr = ref_arr[0]
        
        return map_arr, ref_arr


    @classmethod
    def label_building_candidates(cls, labels, result, tolerance=8):
        """
        Given a gdf of building candidates and a gdf of labels,
        returns gdfs containing correctly-labelled buildings, labels that have matching
        buildings, false negatives (labels w/o buildings) and false positives (buildings
        w/o labels).
        """

        l_coords = [(p.x, p.y) for p in labels.centroid]
        r_coords = [(p.x, p.y) for p in result.centroid]

        matches = {}
        #find building candidates within some tolerance of labelled buildings (true positives)
        #each labelled building will end up with just one matched candidate (if any),
        #even if there are multiple possibilities. This seems OK???
        for idx, coords in enumerate(l_coords):
            for idx2, coords2 in enumerate(r_coords):
                if abs(coords[0] - coords2[0]) <= tolerance\
                and abs(coords[1] - coords2[1]) <= tolerance:
                    matches[idx] = idx2 #key=labelled building, value=candidate building
        matched_candx = result.iloc[list(matches.values())].reset_index(drop=True)
        matched_candx['Values'] = 1

        matched_labels = labels.iloc[list(matches.keys())].reset_index(drop=True)
        matched_labels['Values'] = 1

        #find all the building candidates that don't correspond to labelled buildings
        #false positives
        #temp = [idx for idx, _ in enumerate(r_coords) if idx not in matches.values()]
        #unmatched_candx = result.iloc[temp].reset_index(drop=True)
        #unmatched_candx['Values'] = 0

        #find all the labeled buildings that were not detected
        #false negatives
        #temp = [idx for idx, _ in enumerate(l_coords) if idx not in matches]
        #unmatched_labels = labels.iloc[temp].reset_index(drop=True)
        #unmatched_labels['Values'] = 0

        return matched_candx, matched_labels


    @classmethod
    def rasterize_and_save(cls, gdf, template, out_file, column, fill_nan=True, show=False):
        """
        Given a geodataframe and raference array/metadata, rasterize
        the file using <column> values and save to <out_file>.
        """

        #print(f'  - Rasterizing and saving {out_file}')
        with rasterio.open(template, 'r') as src:
            arr = src.read()
            meta = src.meta

        #TODO: why is nodata not being propagated?
        meta.update({'compress': 'lzw', 'nodata': -9999.0})
        with rasterio.open(out_file, 'w+', **meta) as f:
            shapes = ((geom,value) for geom, value in zip(gdf['geometry'], gdf[column]))
            burned = rasterize(shapes=shapes, fill=0, out_shape=f.shape, transform=f.transform)
            if fill_nan:
                burned[arr[0] < 0] = meta['nodata']
            f.write_band(1, burned)
            
        if show:
            plt.imshow(burned)
            plt.show()
            
            
    @classmethod
    def resample_raster(cls, in_file, out_file, template_file):
        """
        Resample a raster
        """

        with rasterio.open(in_file) as src:
            print(f'Working on {in_file}')

            with rasterio.open(template_file) as template:
                new_height = template.height
                new_width = template.width

            # resample data to target shape
            data = src.read(out_shape=(new_height, new_width),\
                            resampling=Resampling.bilinear)

            # scale image transform
            transform = src.transform * src.transform.scale((src.width / data.shape[-1]),\
                                                        (src.height / data.shape[-2]))
            #update metadata
            profile = src.profile
            profile.update(transform=transform, driver='GTiff', height=new_height,\
                           width=new_width, crs=src.crs, compress='lzw')

        with rasterio.open(out_file, 'w', **profile) as dst:
            dst.write(data)


    def remove_small_buildings(self, outpath, minsize=50):
        """
        Create new version of labelled response files with buildings with area
        <= <minsize> removed. Then, rasterize those files. This doesn't belong
        at all in this module/workflow, but I've resorted to putting it here for
        reasons to do with environments and packages...
        """

        #directory initially containing copies of shapefiles in labeled_region_buildings dir
        path = f'/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_{outpath}'
        files = glob.glob(f'{path}*shp')

        for f in files:

            gdf = gpd.read_file(f)
            gdf = gdf.set_crs(epsg=32605)
            gdf = gdf.loc[gdf.geometry.area > minsize]

            if len(gdf) > 0:
                
                gdf.loc[:, 'FID'] = 1
                gdf.to_file(f, driver='ESRI Shapefile')

                #create a 1m-resolution response tif using the original response tif
                #in the buildings directory as a template
                template = f.replace('shp', 'tif').replace(outpath, 'buildings/')
                self.rasterize_and_save(gdf, template, f.replace('shp', 'tif'), column='FID',\
                                       fill_nan=False)
                
                #create a 2m-resolution response tif using the corresponding ndvi file
                #as a template
                template = f.replace(outpath, 'features/').replace('responses', 'ndvi')
                template = template.replace('shp', 'tif')
                out = f.replace('responses', 'lores_responses').replace('shp', 'tif')
                self.rasterize_and_save(gdf, template, out, column='FID', fill_nan=False)
            else:
                print(f'****Not handling {f}, copy the tif over yourself****')


class MapManips(Utils):
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root, all_labelled_data):
        self.model_output_root = model_output_root
        self.all_labelled_data = all_labelled_data
        self.classified_candidates = None #(defined in classify_all_candidate_buildings)
        Utils.__init__(self, all_labelled_data)


    def probabilities_to_classes(self, applied_model, outfile, threshold_val=0.85,\
                                 nodata_val=-9999, verbose=True):
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
        output[np.logical_and(arr[0] >= threshold_val, output != nodata_val)] = 1

        output = np.expand_dims(output, 0).astype(np.float32)
        meta.update({'count': 1})

        if verbose:
            print(f'  - Writing {outfile}')
            plt.imshow(output[0])
            plt.show()
        with rasterio.open(outfile, 'w', **meta) as f:
            f.write(output)


    def data_cut(self, model_dir, map_type, cut_data_type, threshold, verbose=True):
        """
        Apply an NDVI cut to each map array in model_dir that contains binary classes,
        write to file. Finds the NDVI value for each map pixel, and if it is > ndvi_threshold,
        the class is changed from 1 to 0. Purpose is to to exclude trees that have been
        incorrectly identified as buildings.
        """

        for map_file in glob.glob(f'{model_dir}/*{map_type}*'):

            region = [reg for reg in self.all_labelled_data if reg in map_file][0]

            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()
                meta = f.meta

            #open the corresponding NDVI map
            data, band = list(cut_data_type.items())[0]
            ndvi_file = f'{FEATURE_PATH}{self.all_labelled_data[region]}_{data}.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
            ndvi_arr = ndvi_arr[band:band+1, :, :]

            #set pixels with NDVI > threshold to 0 i.e. not-building; write to file
            if 'ndvi' in data:
                map_arr[ndvi_arr > threshold] = 0
            elif 'irgb' in data:
                map_arr[ndvi_arr < threshold] = 0
            else:
                raise ValueError(f'Not sure how to handle {data} data')
            outfile = map_file.replace(map_type, f"{data.replace('_hires', '')}_cut")
            if verbose:
                print(f"Writing {outfile}")
                plt.imshow(map_arr[0])
                plt.show()
            with rasterio.open(f"{outfile}", 'w', **meta) as f:
                f.write(map_arr)


    def gb_prob_to_raster(self, best_model, bad_bands, xvars, out_prefix, model_dir,\
                                   bnorm, show=False):
        """
        Use an rf model to predict building classes based on xvars and
        save the result to a tif file
        """

        #for each reference region
        for region, data in self.all_labelled_data.items():
            print(region)

            xdata = {}
            
            for n, data_type in enumerate(xvars.keys()):
                if 'refl' in data_type:
                    with rasterio.open(f'{FEATURE_PATH}{data}_{data_type}.tif', 'r') as f:
                        xdata[n] = f.read()
                        xdata[n] = np.delete(xdata[n], bad_bands, axis=0)
                        if bnorm:
                            xdata[n] = xdata[n] / linalg.norm(xdata[n], axis=0)
                elif 'lores_model' in data_type:
                    with rasterio.open(f'{model_dir}{region}_{data_type}.tif', 'r') as f:
                        xdata[n] = f.read()
                        xdata[n] = xdata[n][0:1, :, :]
                else:
                    with rasterio.open(f'{FEATURE_PATH}{data}_{data_type}.tif', 'r') as f:
                        xdata[n] = f.read()
                        meta = f.meta
                        
                if region == 'HBLower' and data_type != 'tch':
                    xdata[n] = xdata[n][:, :, :-1]
                if region == 'HBLower' and data_type != 'cnn_model':
                    xdata[n] = xdata[n][:, :-1, :]
                if region in ['KParadise', 'SKona_B'] and data_type != 'cnn_model':
                    xdata[n] = xdata[n][:, :-1, :-1]
                if region in ['KK1', 'KonaMauka', 'SKona_A'] and data_type != 'cnn_model':
                    xdata[n] = xdata[n][:, :, :-1]

            x_arr = np.concatenate(list(xdata.values()), axis=0)

            #reshape X as necessary, get predicted classes, re-reshape
            original_shape = x_arr[0].shape
            x_arr = np.reshape(x_arr, (x_arr.shape[0], -1)).T
            predicted = best_model.predict_proba(x_arr)
            predicted = predicted[:, 1:2]
            predicted = predicted.T.reshape(original_shape)

            if show:
                plt.imshow(predicted)
                plt.show()
            predicted = np.expand_dims(predicted, axis=0)

            meta.update({'compress': 'lzw', 'count': 1})
            out_file = f"{out_prefix}_{region}.tif"
            with rasterio.open(out_file, 'w', **meta) as f:
                f.write(predicted)


    @classmethod
    def _close_holes(cls, geom):
        """
        Helper method for self.vectorize_and_clean(). Removes interior holes from polygons.
        See stackoverflow.com/questions/63317410/how-to-fill-holes-in-multi-polygons
        -created-when-dissolving-geodataframe-with-ge
        """

        def remove_interiors(poly):
            if poly.interiors:
                return Polygon(list(poly.exterior.coords))
            return poly

        if isinstance(geom, MultiPolygon):
            geos = gpd.GeoSeries([remove_interiors(g) for g in geom])
            geoms = [g.area for g in geos]
            big = geoms.pop(geoms.index(max(geoms)))
            outers = geos.loc[~geos.within(big)].tolist()
            if outers:
                return MultiPolygon([big] + outers)
            return Polygon(big)
        if isinstance(geom, Polygon):
            return remove_interiors(geom)


    def _remove_roads(self, gdf):
        """
        Helper method for self.vectorise_and_clean(). Reads roads from
        ROAD_FILES, applies buffer, removes polygons
        from <gdf> that overlap with buffered roads. Returns edited
        <gdf>.
        """

        for file in ROAD_FILES:
            roads = gpd.read_file(file)
            roads = roads[roads['island'].str.match('Hawaii')]
            roads.geometry = roads.geometry.buffer(1)
            gdf = gdf.loc[~gdf.intersects(roads.unary_union)].reset_index(drop=True)

        return gdf


    def _remove_coastal_artefacts(self, gdf):
        """
        Helper method for self.vectorize_and_clean(). Reads in coastline shapefile
        and removes polygons from <gdf> that intersect with the coast. Same with county
        TML/Parcel map outline. Returns edited <gdf>. Intended for removing coastal
        artefacts.
        """

        #coastline as defined by coastline shapefile
        coast = gpd.read_file(f'{GEO_PATH}Coastline/Coastline.shp')
        coast = coast[coast['isle'] == 'Hawaii'].reset_index(drop=True)
        coast = coast.to_crs(epsg=32605)

        gdf = gdf.loc[gdf.within(coast.unary_union)].reset_index(drop=True)

        #THIS REMOVES SOME BLDGS
        #coastline as defined by County TMK map outline
        #try:
        #    parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_outline.shp')
        #except fiona.errors.DriverError:
        #    parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_-_Hawaii_County.shp')
        #    parcels = parcels.to_crs(epsg=32605)
        #    parcels['dissolve_by'] = 0
        #    parcels = parcels.dissolve(by='dissolve_by').reset_index(drop=True)
        #    parcels = parcels['geometry']
        #    parcels.to_file(f'{GEO_PATH}Parcels/Parcels_outline.shp', driver='ESRI Shapefile')

        #gdf = gdf.loc[gdf.within(parcels.unary_union)].reset_index(drop=True)

        return gdf


    def vectorize_and_clean(self, map_file, out_file, buffers, minpix=25):
        """
        Vectorize a raster map. Then:
        - Remove polygons with area less than <minpix> sq. m
        - Apply a negative buffer to remove spiky pieces and polygons that are connected by
          a 'thread', which are often separate buildings (then rebuffer to approx. original shape)
        - Remove interior holes
        - Clean up coastal artefacts in an individual tile/test region map by vectorizing
          the map and rejecting polygons that don't lie wholly within the coastline
          shapefile
        - Reject polygons that overlap with roads (in State DOT shapefiles).
        - Save the 'cleaned' file in both vector and raster formats, with and without
          being transformed into minimum oriented rectangles
        """

        with rasterio.open(map_file) as f:
            arr = f.read()
            meta = f.meta
            
        #set messed-up edges of regions to NaN, or we may get polygons covering the whole
        #array when we close holes
        arr[:, :4, :] = np.nan
        arr[:, -4:, :] = np.nan
        arr[:, :, :4] = np.nan
        arr[:, :, -4:] = np.nan

        print(f'Vectorizing {map_file}')
        polygons = []
        values = []
        for vec in rasterio.features.shapes(arr.astype(np.int32), transform=meta['transform']):
            polygons.append(shape(vec[0]))
            values.append(vec[1])
        gdf = gpd.GeoDataFrame(crs=meta['crs'], geometry=polygons)

        gdf['Values'] = values

        #remove polygons composed of NaNs or 0s (building == 1)
        gdf = gdf[gdf['Values'] > 0]

        #Apply negative then positive buffer, to remove spiky bits and disconnect
        #polygons that are connected by a thread (they are usually separate buildings)
        gdf.geometry = gdf.geometry.buffer(buffers[0])
        #Delete polygons that get 'emptied' by negative buffering
        gdf = gdf[~gdf.geometry.is_empty]
        #Apply positive buffer to regain approx. original shape
        gdf.geometry = gdf.geometry.buffer(buffers[1])
        #If polygons have been split into multipolygons, explode into separate rows
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)
        
        #Fill any interior holes
        gdf.geometry = gdf.geometry.apply(lambda row: self._close_holes(row))

        #remove any remaining too-small polygons
        if minpix is not None:
            gdf = gdf.loc[gdf.geometry.area >= minpix]

        #Remove polygons that intersect roads (do before getting any bounding rectangle)
        print('  - Removing polygons that intersect with roads')
        gdf = self._remove_roads(gdf)

        #Remove polygons that are outside the County parcel/TMK map outline, or outside the
        #coastline
        print('  - Removing polygons outside/overlapping the coast')
        gdf = self._remove_coastal_artefacts(gdf)
        
        #Save two versions of the final maps, each as both shapefiles and rasters
        #This first one has no bounding boxes applied; the polygons will be irregular
        #First, save (create a copy of) the geodataframe as it is
        gdf_bounds = gdf.copy()
        if len(gdf) > 0:
                gdf.geometry = gdf.geometry.buffer(buffers[2])
                self.rasterize_and_save(gdf, map_file, out_file, 'Values', show=False)
                gdf.to_file(out_file.replace('.tif', '.shp'), driver='ESRI Shapefile')

        #In this second one, the buildings are replaced by their minimum oriented bounding boxes
        def get_bbox(geom):
            poly = Polygon(geom)
            return poly.minimum_rotated_rectangle
        
        def buffer_bbox(geom):
            area = geom['geometry'].area
            buffer_distance = area * -0.0035
            buffered_geom = geom['geometry'].buffer(buffer_distance)
    
            if buffered_geom.is_empty:
                return geom['geometry']
            else:
                return buffered_geom

        #replace geometries with minimum oriented bounding boxes
        gdf_bounds['geometry'] = gdf_bounds.geometry.apply(get_bbox)
        #boxes are too big, though, so decrease size a bit
        gdf_bounds['geometry'] = gdf_bounds.apply(buffer_bbox, axis=1)

        #Write the bounding box files
        if len(gdf_bounds) > 0:
            out_file = out_file.replace('poly', 'bounds')
            print(out_file)
            self.rasterize_and_save(gdf_bounds, map_file, out_file, 'Values', show=False)
            gdf_bounds.to_file(out_file.replace('.tif', '.shp'), driver='ESRI Shapefile')


class Ensemble(MapManips):
    """
    Methods for creating ensemble maps
    """

    def __init__(self, model_output_root, test_sets, ensemble_path, all_labelled_data):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.ensemble_path = ensemble_path
        MapManips.__init__(self, model_output_root, all_labelled_data)
        
        
    def average_probabilities(self, model_nums, model_kind, region, show=True):
        """
        Create an ensemble map for a region by taking the average of the
        probability maps given by <model_nums>. Ensemble maps are
        saved as tifs.
        """

        print(f'Working on {region}')
        model_list = []

        #get the relevant maps, for each region
        for num in model_nums:
            if model_kind == 'CNN':
                this_file = f'{self.model_output_root}combo_{num}/applied_model_{region}.tif'
            elif model_kind == 'GB':
                this_file = f'{self.ensemble_path}gb_prob_run{num}_{region}.tif'
            else:
                raise ValueError('<model_kind> must be one of "CNN"|"GB"')
            with rasterio.open(this_file, 'r') as f:
                this_array = f.read()
                meta = f.meta
            model_list.append(this_array)

        #find the mean of the maps
        combo = np.mean(model_list, axis=0)
        if model_kind == 'CNN':
            combo = combo[1:2, :, :]

        if show:
            plt.imshow(combo[0][20:-20, 20:-20])
            plt.show()
            
        #write the ensemble map to file
        os.makedirs(self.ensemble_path, exist_ok=True)
        meta.update({'count': 1})
        if model_kind == 'CNN':
            with rasterio.open(f"{self.ensemble_path}mean_probability_{region}.tif", 'w',\
                               **meta) as f:
                f.write(combo)
        elif model_kind == 'GB':
            with rasterio.open(f"{self.ensemble_path}gb_ensemble_prob_{region}.tif", 'w',\
                               **meta) as f:
                f.write(combo)


    def create_ensemble(self, model_nums, region, map_type='threshold', show=True):
        """
        Create an ensemble map for a region, starting from a threshold,
        applied, or ndvi_cut map, by taking the sum of each map pixel.
        This means that the value of each pixel in the output map indicates the number
        of input maps that classified that pixel as a building. Ensemble maps are
        saved as tifs.
        """

        allowed_types = ['threshold', 'applied', 'ndvi_cut', 'amcu_cut']
        if map_type not in allowed_types:
            raise ValueError(f"<map_type> must be one of {allowed_types}")

        print(f'Working on {region}')
        model_list = []

        #get the relevant maps, for each test region
        for num in model_nums:
            this_file = f'{self.model_output_root}combo_{num}/{map_type}_model_{region}.tif'
            with rasterio.open(this_file, 'r') as f:
                this_array = f.read()
                meta = f.meta
            model_list.append(this_array)

            #find the SUM of the maps, which can be interpreted as the number of 'votes'
            #for a candidate building pixel
            combo = np.sum(model_list, axis=0)

        if show:
            plt.imshow(combo[0][20:-20, 20:-20])
            plt.show()

        #write the ensemble map to file
        os.makedirs(self.ensemble_path, exist_ok=True)
        meta.update(count=1)
        with rasterio.open(f"{self.ensemble_path}{map_type}_ensemble_{region}.tif", 'w',\
                               **meta) as f:
            f.write(combo)


    def choose_cut_level(self, ensemble_file, n_votes, out_file, show=True):
        """
        For an ensemble created by self.create_ensemble() return a version converted to binary
        classes using <n_votes>. The input ensemble will have pixel values = 0-n in steps of 1,
        where n is the number of models that went into the ensemble. For example, an ensemble map
        created from 5 individual models has pixel values like this:
        0 - no models assigned class 'building' to this pixel
        1 - 1 model assigned class 'building' to this pixel
        2 - 2 models assigned class 'building' to this pixel
        3 - 3 models assigned class 'building' to this pixel
        4 - 4 models assigned class 'building' to this pixel
        5 - 5 models assigned class 'building' to this pixel
        Setting <n_votes>=4 will produce a model with binary building/not-building classes such
        that each building will have received a positive classification from 4 input models.
        This method must be used in order to produce a model that is understood by Performance.
        performance_stats.
        """

        input_name = f"{self.ensemble_path}{ensemble_file}.tif"
        output_name = f"{self.ensemble_path}{out_file}.tif"

        with rasterio.open(input_name, 'r') as f:
            arr = f.read()
            meta = f.meta

        arr[arr < n_votes] = 0
        arr[arr >= n_votes] = 1

        if show:
            plt.imshow(arr[0])
            plt.show()

        with rasterio.open(output_name, 'w', **meta) as f:
            f.write(arr)


class Evaluate(Utils):
    """
    Methods for evaluating model performance and map quality and characteristics
    """

    def __init__(self, model_output_root, training_sets, all_labelled_data):
        self.model_output_root = model_output_root
        self.training_sets = training_sets
        self.all_labelled_data = all_labelled_data
        Utils.__init__(self, all_labelled_data)


    @classmethod
    def _histo(cls, ax, data, bins, xtext, xlims, xlinepos=None, title='', legend=False):
        """
        Helper method for self.probability_hist. Puts the data into
        the histogram and does some axis stuff.
        """

        counts, _, _ = ax.hist(data[0], bins, histtype='stepfilled', ec='k', fc='0.7',\
                              label='True building')
        ax.hist(data[1], bins, histtype='step', ec='0.4', label='Not building')
        if xlinepos is not None:
            ax.axvline(xlinepos, color='k', ls='--')
        if legend:
            ax.legend(loc='lower left')
        ax.set_title(title)
        ax.set_xlim(xlims)
        ax.set_xlabel(xtext)
        ax.set_ylim(0, max(counts)*1.05)
        ax.set_ylabel('Frequency')


    def probability_hist(self, model_dir, map_type, threshold, title, legend):
        """
        Produces a histogram of probability values for CNN or GB maps of training+test
        regions. Shows building and not-building pixels, according to the labelled
        reference files.
        """

        print(f'Working on {map_type} maps from {model_dir}')

        bldg_prob = []
        notbldg_prob = []

        #for each test_region map
        for map_file in glob.glob(f'{model_dir}/*{map_type}*'):
            region = [reg for reg in self.all_labelled_data if reg in map_file][0]
            
            #quick hack to deal with my stupid filenames
            if 'Hamakua_A' in map_file and 'Hamakua' in region:
                continue

            #open the map and response files
            try:
                map_arr, ref_arr = self._get_map_and_ref_data(map_file, region, 'lores_')
            except rasterio.errors.RasterioIOError:
                #this is KonaMauka and CCTrees
                with rasterio.open(map_file, 'r') as f:
                    map_arr = f.read()
                map_arr = map_arr[0]
                ref_arr = np.zeros_like(map_arr)

            if region == 'HBLower' and map_type != 'lores_model':
                ref_arr = ref_arr[:, :-1]

            #make arrays of candidates that are and aren't buildings, based on labeled refs
            bldg = np.where((ref_arr == 1), map_arr, -9999)
            notbldg = np.where((ref_arr == 0), map_arr, -9999)

            bldg_prob.append([x for x in bldg.flatten().tolist() if x > -9999])
            notbldg_prob.append([x for x in notbldg.flatten().tolist() if x > -9999])

        bldg_prob = [x for y in bldg_prob for x in y]
        notbldg_prob = [x for y in notbldg_prob for x in y]
        
        if map_type == 'lores_model':
            #this should really be dealt with elsewhere but it's too late to refactor
            #everything now...
            bldg_prob = [1 - x for x in bldg_prob]
            notbldg_prob = [1 - x for x in notbldg_prob]

        _, ax = plt.subplots(1, 1, figsize=[6, 6])
        bins = [x * 0.01 for x in range(0, 101, 1)]
        self._histo(ax, [bldg_prob, notbldg_prob], bins,\
                    'Probability of belonging to building class',\
                    [0, 1], threshold, title, legend)
        plt.savefig(f'/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/for_figures/{map_type}_hist.png',\
                    dpi=450)
        

    def get_vars_from_rasters(self, xvars, bad_bands, model_dir, run_id, mask_shade=True,\
                              bnorm=True):
        """
        Reads in building raster files in training set, gets XXX for each pixel,
        reformats as X, y variables for RF fitting with self.ml_classify.
        Trims...
        """

        y = []
        X = [[] for _ in range(sum([len(x) for x in xvars.values()]))]

        #for each reference region
        for region, data in self.training_sets.items():
            print(f'Getting data for {region}')
            
            n = 0
            
            #get the shade mask, if needed
            if mask_shade:
                with rasterio.open(f'{FEATURE_PATH}{data}_pure_shade.tif') as f:
                    shade = f.read()
                    shade = shade[:, 20:-20, 20:-20]

            #open the X (predictor) variables
            data_types = [x for x in xvars.keys()]
            data_types.append('reference')
            for data_type in data_types:

                if data_type == 'refl':
                    #TODO: make it so that IRGB data can be handled and used for NaN mask
                    with rasterio.open(f'{FEATURE_PATH}{data}_{data_type}.tif', 'r') as f:
                        arr = f.read()
                        arr = arr[:, 20:-20, 20:-20]
                        #remove bands affected by water vapour
                        arr = np.delete(arr, bad_bands, axis=0)
                        if bnorm:
                            arr = arr / linalg.norm(arr, axis=0)
                            if region == 'HBLower':
                                #fix brightness norm-induced NaNs in HBLower
                                arr = np.nan_to_num(arr)
                        #make a mask where pixels with no spectroscopy data are identified as NaN
                        #nan_mask = np.where(arr[0] < 0, np.nan, arr[0])
                        #nan_mask = np.isnan(nan_mask)
                elif data_type == 'lores_model':
                    with rasterio.open(f'{model_dir}{region}_{data_type}.tif', 'r') as f:
                        arr = f.read()
                        arr = arr[0:1, 20:-20, 20:-20]
                elif data_type == 'tch':
                    with rasterio.open(f'{FEATURE_PATH}{data}_{data_type}.tif', 'r') as f:
                        arr = f.read()
                        arr = arr[:, 20:-20, 20:-20]
                elif data_type == 'reference':
                    with rasterio.open(f'{RESPONSE_PATH}{data}_lores_responses.tif') as f:
                        arr = f.read()
                        arr = arr[:, 20:-20, 20:-20]
                else:
                    raise ValueError('Data types can only include refl|lores_model|tch')

                #set pixels with no spectroscopy data to NaN
                #for some reason HBLower tch and shade mask have 1 fewer row than refl
                #and model, so we need to do a little reshaping here
                if region == 'HBLower' and data_type != 'tch':
                    arr = arr[:, :, :-1]

                #Get data into 1D so can apply shade mask i.e. remove shaded pixels
                if data_type == 'reference':
                    arr = np.ravel(arr)
                    if mask_shade:
                        arr = np.where(np.ravel(shade)<255, np.nan, arr)
                    arr = arr[~np.isnan(arr)]
                    y.extend(arr)
                else:
                    arr = np.reshape(arr, (arr.shape[0], -1))
                    for band, info in enumerate(arr):
                        if mask_shade:
                            info = np.where(np.ravel(shade)<255, np.nan, info)
                        info = info[~np.isnan(info)]
                        X[band+n].extend(info)
                    n += arr.shape[0]
                    
        #reformat as needed by sklearn for RF fitting
        print('Reformatting X...')
        X = np.array(X).T
        print('Reformatting y...')
        y = np.array(y)
        print(X.shape, y.shape)
        
        print(f'Saving {run_id} X and y files')
        with open(f'{model_dir}X_{run_id}.pkl', "wb") as f:
            pickle.dump(X, f, protocol=4)
            
        with open(f'{model_dir}y_{run_id}.pkl', "wb") as f:
            pickle.dump(y, f, protocol=4)

        return X, y


    @classmethod
    def sample_from_vars(cls, X, y, sample=10000):
        """
        Divide large X and y arrays into arrays including only elements
        where y=0 or y=1, then take a random sample from each of those. Doing it
        that way means that buildings are better represented than
        they would be in a simple random sample from all elements. Sampling is
        needed because the original arrays are too large to fit models to.
        """

        def sample_zeros_or_ones(value):
            temp_y = y[y==value]
            temp_x = X[y==value]
            print(f'There are {len(temp_x)} pixels with true class = {value}')
            try:
                idx = np.random.choice(np.arange(temp_y.shape[0]), int(sample/2), replace=False)
                temp_y = temp_y[idx]
                temp_x = temp_x[idx]
            except ValueError:
                print(f'{len(temp_x)} < {sample}; returning all pixels')

            return temp_x, temp_y

        #sample <sample>/2 X and y that are true positives
        #real buildings: ref_array=1 and map_array=1 (latter comes out of get_Xy_from_rasters)
        ones_x, ones_y = sample_zeros_or_ones(value=1)
        print(f' -- Sampled {len(ones_x)} pixels')

        #sample <sample>/2 X and y that are false positives
        #ref_arry=0 even though map_array=1
        zeros_x, zeros_y = sample_zeros_or_ones(value=0)
        print(f' -- Sampled {len(zeros_x)} pixels')

        y = np.concatenate((ones_y, zeros_y), axis=0)
        X = np.concatenate((ones_x, zeros_x), axis=0)

        return X, y


    def ml_classify(self, X, y, save_to, n_iter=50, scoring='recall'):
        """
        Fit XGBoost models to the building candidates in self.classified candidates, which
        are labelled 0 or 1 depending on whether they are false positives or genuine buildings.
        Tune the n_estimators and max_depth hyperparameters...
        """

        #split the data into training and test sets
        print('Creating training and test data')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y,\
                                                  random_state=RANDOM_STATE)
        
        #save the training and test data for calculating metrics and making figures
        save_me = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        with open(save_to.replace('gb', 'train_test'), "wb") as f:
            pickle.dump(save_me, f)

        #start the parameter search and fitting
        print(f'Fitting model to dataset with shape={X_train.shape}')

        clf = XGBClassifier(tree_method='gpu_hist', gpu_id=0)
        #parameter tuning suggestions from
        #machinelearningmastery.com/configure-gradient-boosting-algorithm/
        params = {'n_estimators': [500],\
                  'eta': [0.01, 0.015, 0.025, 0.05, 0.1],\
                  'max_depth': [4, 6, 8, 10],\
                  'min_child_weight': [1, 10, 100],\
                  'colsample_bytree': [0.6, 0.8, 1.0]}

        cv = StratifiedKFold(n_splits=5)
        rand_search = RandomizedSearchCV(clf, param_distributions=params, scoring=scoring,\
                                         n_iter=n_iter, cv=cv, n_jobs=1, verbose=2,\
                                         random_state=RANDOM_STATE)

        #fit the random search object to the data (this actually does the work)
        rand_search.fit(X_train, y_train)

        #save the model
        print('Writing file')
        with open(save_to, "wb") as f:
            pickle.dump(rand_search, f)

        return rand_search


    @classmethod
    def ml_metrics(cls, model, xvars, xy_file, save_to):
        """
        """
        
        #print the best hyperparameters
        print('Best score:',  model.best_score_)
        print('Best hyperparameters:',  model.best_params_)

        _, ax = plt.subplots(1, 1, figsize=(12, 4))
        scores = model.cv_results_['mean_test_score']
        plt.plot(range(len(scores)), scores)
        for x, text in enumerate(model.cv_results_['params']):
            txt = [t for t in text.values()]
            ax.text(x, 0.81, txt, rotation=90, size=8)
        plt.ylim(0.8, 0.9)
        plt.show()
        
        with open(xy_file, "rb") as f:
            xy = pickle.load(f)
        X_train = xy['X_train']
        X_test = xy['X_test']
        y_test = xy['y_test']
        #split the (training) data into training and test sets
        #print('Splitting data and making predictions')
        #X_train, X_test, _, y_test = train_test_split(X, y, test_size=0.5, stratify=y,\
        #                                        random_state=RANDOM_STATE)
        
        #predict classes for test data
        y_pred = model.predict(X_test)

        #show the classification report
        print('Performance on test portion of training regions')
        print(classification_report(y_test, y_pred))

        #show the feature importances
        xnames = [x for y in xvars.values() for x in y]
        _ = plt.figure()

        feature_importances = pd.Series(model.best_estimator_.feature_importances_,\
                                        index=xnames).\
                                        sort_values(ascending=False)
        print('Most important features')
        display(feature_importances.head(10).to_frame().rename(columns={0: 'Importance'}).T)
        print('Least important features')
        display(feature_importances.tail(10).to_frame().rename(columns={0: 'Importance'}).T)
        
        #Shapley values and plots
        print('Getting SHAP values')
        model = model.best_estimator_
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap_obj = explainer(X_train)
        
        print('Creating bar plot')
        shap.plots.bar(shap_obj, max_display=12, show=True)
        
        print('Creating summary plot')
        xnames = [x for y in xvars.values() for x in y]
        shap.summary_plot(shap_values,
                             features=X_train,
                             feature_names=xnames,
                             show=True,
                             plot_size=(30,15))
        
        print('Writing SHAP values to file')
        with open(save_to, "wb") as f:
            pickle.dump(shap_values, f)


class Stats(Utils):
    """
    Methods for calculating and plotting performance statistics (map quality stats)
    """

    def __init__(self, model_output_root, test_sets, all_labelled_data, analysis_path):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.analysis_path = analysis_path
        self.matched_candx = None #defined in self.vector_stats
        self.matched_labels = None
        self.unmatched_labels = None
        self.unmatched_candx = None
        Utils.__init__(self, all_labelled_data)


    @classmethod
    def _boundary_shp_to_mask(cls, boundary_file, background_file):
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


    def raster_stats(self, model_dir, map_kind, regions, resolution=''):
        """
        Given a model class prediction array and a set of responses, calculate precision,
        recall, and f1-score for each class (currently assumes binary classes)
        """

        stats_dict = {}

        #get the list of map files
        map_list = glob.glob(f'{model_dir}/*{map_kind}*tif')

        #for each test_region map
        for map_file in map_list:
            
            #quick hack to deal with my stupid filenames
            if 'Hamakua_A' in map_file and 'Hamakua' in regions.keys():
                continue

            #is this a test region map? Or a training region map?
            res = [region for region in regions.keys() if region in map_file]
            #if this isn't a test region map, we'll exit and not calculate stats for it
            if len(res) == 0:
                continue

            test_region = res[0]
            if test_region in ['KonaMauka', 'CCTrees']:
                continue

            #get the map and labelled response files for this test region
            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region, resolution)

            #get the boundary mask (requires response file as opposed to array)
            ref_file = f'{RESPONSE_PATH}{regions[test_region]}_{resolution}responses.tif'
            boundary_file = f'{BOUNDARY_PATH}{regions[test_region]}_boundary.shp'

            #create an array of the same shape as the test region
            #in which everything outside the test dataset boundary/boundaries is NaN
            masko = self._boundary_shp_to_mask(boundary_file, ref_file)
            if test_region == 'HBLower':
                masko = masko[:, :-1]
                ref_arr = ref_arr[:, :-1]
            
            #trim outer 20 pix off each array
            #(map edges are NaN; npix to trim should really be related to window size)
            map_arr = map_arr[20:-20, 20:-20]
            ref_arr = ref_arr[20:-20, 20:-20]
            masko = masko[20:-20, 20:-20]

            # insert the labelled responses into the array, inside the training boundaries
            masko[ref_arr != 0] = ref_arr[ref_arr != 0]

            # flatten to 1D and remove NaNs
            predicted = map_arr.flatten()
            expected = masko.flatten()
            predicted = list(predicted[~(np.isnan(expected))])
            expected = list(expected[~(np.isnan(expected))])

            # get performance metrics
            stats_dict[test_region] = classification_report(expected, predicted, output_dict=True)
            stats_dict[test_region].update({'jaccard': jaccard_score(expected, predicted)})

        return stats_dict


    @classmethod
    def display_raster_stats(cls, stats_dict):
        """
        Display a data frame containing the stats obtained by self.raster_stats.
        """

        df = pd.DataFrame()
        for item in stats_dict:
            if item not in ['KonaMauka', 'CCTrees']: #exclude regions with no real buildings
                df.loc[item, 'Precision'] = stats_dict[item]['1.0']['precision']
                df.loc[item, 'Recall'] = stats_dict[item]['1.0']['recall']
                df.loc[item, 'F1-score'] = stats_dict[item]['1.0']['f1-score']
                df.loc[item, 'IoU'] = stats_dict[item]['jaccard']
        df.sort_index(inplace=True)
        df.loc['Mean'] = df.mean()
        display(df)


    def plot_raster_stats(self, stats_dict, plot_file):
        """
        Make a multi-panel plot in which each panel shows precision, recall, and f1-score
        for the test areas for a single model. At least as the models were orginally
        set up, rows=data type (DSM, eigenvalues, hillshade), columns=window size
        (16, 32, 64).
        """

        rows = int(np.ceil(len(stats_dict) / 3))
        fig, _ = plt.subplots(rows, 3, figsize=(12, 4*rows))
        n = 0
        for model, ax in zip(sorted(stats_dict.keys()), fig.axes):
            precision = []
            recall = []
            f1score = []
            regions = []
            iou = []
            for region, stats in sorted(stats_dict[model].items()):
                if region not in ['KonaMauka', 'CCTrees']:
                    precision.append((stats['1.0']['precision']))
                    recall.append((stats['1.0']['recall']))
                    f1score.append((stats['1.0']['f1-score']))
                    iou.append(stats['jaccard'])
                    regions.append(region)
            x = range(len(precision))
            ax.plot(x, precision, color='b', ms=6, marker='o', label='precision')
            ax.plot(x, recall, color='r', ms=6, marker='o', label='recall')
            ax.plot(x, f1score, color='k', ms=6, marker='o', label='f1-score')
            ax.plot(x, iou, color='limegreen', ms=6, marker='o', label='IoU')
            ax.axhline(0.8, color='0.5', ls='--')
            if n == 0:
                ax.legend(loc='lower left')
            ax.set_ylim(0, 1)
            ax.set_xticks(x, regions, rotation=45, ha='right')
            if 'ensemble' in plot_file:
                ax.set_title(f'Ensemble {model}votes')
            else:
                ax.set_title(f'Model #{model}')
            n += 1

        for n, ax in enumerate(fig.axes):
            if n >= len(stats_dict):
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.analysis_path+plot_file, dpi=400)

        
    def match_buildings_to_labels(self, model_dir, map_kind):
        """
        """
        
        self.matched_candx = pd.DataFrame()
        self.matched_labels = pd.DataFrame()

        map_list = glob.glob(f'{model_dir}*{map_kind}*shp')

        #for each test_region map
        for map_file in map_list:

            #test regions only
            res = [region for region in self.test_sets.keys() if region in map_file]
            if len(res) == 0:
                continue
            test_region = res[0]
            
            labels = gpd.read_file(f"{RESPONSE_PATH}{self.test_sets[test_region]}_responses.shp")
            result = gpd.read_file(f'{model_dir}{map_kind}_{test_region}.shp')
            
            #Link labels to mapped buildings
            matched_candx, matched_labels, = \
                                    self.label_building_candidates(labels=labels, result=result)
            matched_candx['Region'] = test_region
            matched_labels['Region'] = test_region

            #Calculate IoU for each test region
            #iou = []
            #for idx, row in matched_labels.iterrows():
            #    poly1 = row['geometry']
            #    poly2 = matched_candx.loc[idx, 'geometry']
            #    intersect = poly1.intersection(poly2).area
            #    union = poly1.union(poly2).area
            #    iou.append(intersect / union)
            #tempdict['iou'] = 'N/A'#np.mean(iou)
            #print('***PROBABLY NEED TO EDIT IOU CALC TO INCLUDE FALSE POSITIVES/NEGATIVES***')

            #Write out files for diagnostic purposes
            for gdf, name in zip ([matched_labels, matched_candx],\
                                  ['matched_labels', 'matched_candidates']):
                if len(gdf) > 0:
                    #Sometimes there are no unmatched X so no point saving
                    gdf.to_file(f'{model_dir}matches/{test_region}_{name}.shp',\
                                driver='ESRI Shapefile')

            self.matched_labels = pd.concat([self.matched_labels, matched_labels])
            self.matched_candx = pd.concat([self.matched_candx, matched_candx])
            


    def vector_stats(self, model_dir, map_kind):
        """
        Given a shapefile of building candidates and a labelled response file,
        calculate precision, recall, f1-score
        """

        stats_df = pd.DataFrame()
        self.true_positives = gpd.GeoDataFrame(geometry=[], crs='epsg:32605')
        self.false_negatives = gpd.GeoDataFrame(geometry=[], crs='epsg:32605')
        self.false_positives = gpd.GeoDataFrame(geometry=[], crs='epsg:32605')
        #all labelled buildings regardless of whether or not they are detected
        self.all_labelled_buildings = gpd.GeoDataFrame(geometry=[], crs='epsg:32605')

        #get the list of map files
        map_list = glob.glob(f'{model_dir}*{map_kind}*shp')

        #for each test_region map
        for map_file in map_list:

            #is this a test region map? Or a training region map?
            res = [region for region in self.test_sets.keys() if region in map_file]

            #if this isn't a test region map, we'll exit and not calculate stats for it
            if len(res) == 0:
                continue
                
            if res[0] in ['KonaMauka', 'CCTrees']:
                continue

            test_region = res[0]

            labels = gpd.read_file(f"{RESPONSE_PATH}{self.test_sets[test_region]}_responses.shp")
            result = gpd.read_file(f'{model_dir}{map_kind}_{test_region}.shp')
            boundary = gpd.read_file(f'{BOUNDARY_PATH}{self.test_sets[test_region]}_boundary.shp')
            
            #use only results that are within region boundaries, or will have too many false
            #positives. Also, buffer region boundaries by 20 m, as for raster stats
            #TODO: raster stats were clipped by 20 pix at each side, is that 20 m or 40 m?
            boundary.geometry = boundary.geometry.buffer(-20)
            result = result[result.apply(lambda row: row.geometry.intersects(boundary.geometry[0]), axis=1)]
            
            #Find labelled buildings that have a corresponding (overlapping) polygon in the map
            # --> true positives
            true_positives = []
            for index1, row1 in labels.iterrows():
                for index2, row2 in result.iterrows():
                    if row1.geometry.intersects(row2.geometry):
                        true_positives.append(row1.geometry)
                        break
            self.true_positives = self.true_positives.append(gpd.GeoDataFrame(geometry=true_positives),\
                                                             ignore_index=True)

            #Find labelled buildings that don't
            # --> false negatives
            false_negatives = []
            for index1, row1 in labels.iterrows():
                is_disjoint = True
                for index2, row2 in result.iterrows():
                    if not row1.geometry.disjoint(row2.geometry):
                        is_disjoint = False
                        break
                if is_disjoint:
                    false_negatives.append(row1.geometry)
            self.false_negatives = self.false_negatives.append(gpd.GeoDataFrame(geometry=false_negatives),\
                                                             ignore_index=True)
                    
            #and do the reverse to find false positives
            false_positives = []
            for index1, row1 in result.iterrows():
                is_disjoint = True
                for index2, row2 in labels.iterrows():
                    if not row1.geometry.disjoint(row2.geometry):
                        is_disjoint = False
                        break
                if is_disjoint:
                    print(row1.geometry.area)
                    false_positives.append(row1.geometry)
            self.false_positives = self.false_positives.append(gpd.GeoDataFrame(geometry=false_positives),\
                                                             ignore_index=True)

            self.all_labelled_buildings = pd.concat([self.all_labelled_buildings, labels])

            #Calculate precision, recall and f1-score for each test region
            #precision = TP/(TP+FP)
            #recall = TP/(TP+FN)
            tempdict = {}
            tempdict['precision'] = len(true_positives)/(len(true_positives)\
                                                             +len(false_positives))
            tempdict['recall'] = len(true_positives)/(len(true_positives)\
                                                          +len(false_negatives))
            tempdict['f1-score'] = hmean([tempdict['precision'], tempdict['recall']])

            for name, stat in zip(list(tempdict.keys()), list(tempdict.values())):
                stats_df.loc[test_region, name] = stat
                
        stats_df.sort_index(inplace=True)
        stats_df.loc['Mean'] = stats_df.mean()
        display(stats_df)
        

    def distance_to_closest(self, gdf):
        """
        """
            
        min_distances = []
        for i, row in gdf.iterrows():
            distances = []
            for j, other_row in self.all_labelled_buildings.iterrows():
                distance = row['geometry'].distance(other_row['geometry'])
                if distance > 0.001: #exclude same polygon in other df
                    distances.append(distance)
            min_distance = min(distances)
            min_distances.append(min_distance)

        return min_distances


    def building_size_plots(self, sep=6):
        """
        Create a multi-part figure showing (1) detection rate vs building size, and
        (2) mapped size vs manual size
        """
        
        plt.rcParams.update({'font.size': 14})

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        #TODO: break up this big method
        #----------------------------------------------------------
        #HISTOGRAMS OF BUILDING SIZES (ALL, DETECTED)

        step = 20
        bins = np.arange(50, 1020, step)
        ax1.hist(self.all_labelled_buildings.geometry.area, bins=bins,\
                 label='All labeled buildings', color='0.7')
        ax1.hist(self.true_positives.geometry.area, bins=bins,\
                             color='k', histtype='step', lw=1,\
                             label='Detected buildings')

        ax1.set_xlim(0, 900)
        ax1.set_ylim(0, 105)

        #TWINX: RECALL PLOT
        ax1.set_xlabel('Manually-outlined building size, sq. m')
        ax1.set_ylabel('Number of buildings')
        
        ax1b = ax1.twinx()
        tp, _, _ = ax1b.hist(self.true_positives.geometry.area, bins=bins, alpha=0)
        fn, _, _ = ax1b.hist(self.false_negatives.geometry.area, bins=bins, alpha=0)
        fp, _, _ = ax1b.hist(self.false_positives.geometry.area, bins=bins, alpha=0)

        precision = tp / (tp + fp)
        ax1b.plot(bins[:len(precision)]+step/2, precision, color='k', marker='*', mfc='w', ms='14',\
                  ls='-', label='Precision')

        recall = tp / (tp + fn)
        ax1b.plot(bins[:len(recall)]+step/2, recall, color='k', marker='o', mfc='w', ms='10',\
                  ls='-', label='Recall')
        
        df = pd.DataFrame()
        df['Bin start'] = bins[:-1]
        df['Recall'] = recall
        df['Precision'] = precision
        display(df.head(15))
        for n in [0.25, 0.5, 0.75]:
            quantile = self.all_labelled_buildings.geometry.area.quantile(n)
            print(f'The {n} quantile of building area is {quantile:.0f} sq m')
        small_bldgs = self.all_labelled_buildings[self.all_labelled_buildings.geometry.area < 90]
        pct = (len(small_bldgs) / len(self.all_labelled_buildings)) * 100
        print(f'{pct:.0f}% of labeled buildings have area < 90 sq m')

        ax1b.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines1b, labels1b = ax1b.get_legend_handles_labels()
        ax1b.legend(lines1 + lines1b, labels1 + labels1b, loc='center right')
        ax1b.set_ylabel('Recall, precision')
        ax1.text(0.03, 0.96, '(a)', transform=ax1.transAxes)

        #----------------------------------------------------------
        #LABELLED BUILDING SIZE VS MODELED BUILDING SIZE
        data = pd.DataFrame()
        data['Manual size'] = self.matched_labels.geometry.area
        data['Model size'] = self.matched_candx.geometry.area
        data['Region'] = self.matched_candx['Region']

        diff = data['Manual size'] - data['Model size']
        diff = diff.abs() / data['Manual size']
        print(f'Median absolute error on building size = {diff.median()*100:.0f}%')

        #find closest building to each building in manually-labelled data
        min_distances = self.distance_to_closest(self.matched_labels)
        data['Closest neighbour distance'] = min_distances

        #plot buildings with close neighbours in red
        plot_me = data[data['Closest neighbour distance'] <= sep]
        sns.scatterplot(data=plot_me, x='Manual size', y='Model size', ax=ax2, marker='*',\
                            ec='k', fc='w', label=f'Buildings with neighbor within {sep} m')
        #plot all other buildings
        plot_me = data[data['Closest neighbour distance'] > sep]
        sns.scatterplot(data=plot_me, x='Manual size', y='Model size', ax=ax2, color='k',\
                        alpha=0.5, label='All other buildings')

        #1:1 line
        ax2.plot([0, 1000], [0, 1000], ls='--', color='k')

        #r, p_val = pearsonr(data['Manual size'], data['Model size'])
        #ax2.text(0.85, 0.05, f'r = {r:.2f}', transform=ax2.transAxes)
        ax2.text(0.03, 0.96, '(b)', transform=ax2.transAxes)
        
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 1000)
        ax2.set_xlabel('Manually-outlined building size, sq. m')
        ax2.set_ylabel('Modeled building size, sq. m')

        plt.tight_layout()
        plt.savefig(f'{FIGURE_OUTPUT_PATH}buildings.png', dpi=450)


class MLClassify(MapManips):
    """
    Methods for classifying building polygons - moved here for now because
    classifying at the raster stage is probably better; will probably
    remove.
    """

    def __init__(self, model_output_root, training_sets, all_labelled_data):
        self.training_sets = training_sets
        MapManips.__init__(self, model_output_root, None, all_labelled_data)


    def classify_all_candidate_buildings(self, model_dir):
        """
        Create a gdf, self.classified candidates, containing polygons that are genuine
        buildings (Value=1) and false positives (Value=0), respectively. Add columns
        containing the mean aMCU (all bands) for each candidate.
        """

        self.classified_candidates = gpd.GeoDataFrame()

        #get the list of map files
        map_list = sorted(glob.glob(f'{model_dir}/*cleaner*.tif'))

        #for each map
        for map_file in map_list:

            #get the name of the region
            region = [reg for reg in self.all_labelled_data.keys() if reg in map_file][0]
            print(region)

            #read the map as a shapefile, containing 1 polygon per building candidate
            map_gdf = gpd.read_file(map_file.replace('.tif', '.shp'))

            #read the labelled response file for the test area, also as shapefile
            labels_gdf = gpd.read_file\
                        (f'{RESPONSE_PATH}{self.all_labelled_data[region]}_responses.shp')

            #open the aMCU file for the test region
            amcu_file = f'{FEATURE_PATH}{self.all_labelled_data[region]}_amcu_hires.tif'
            with rasterio.open(amcu_file, 'r') as f:
                affine = f.transform
                amcu_arr = f.read()

            #find building candidates that have been correctly and incorrectly identified
            correct, incorrect, _, _ = self.label_building_candidates(labels_gdf, map_gdf)
            correct['Region'] = region
            incorrect['Region'] = region

            def poly_stats(building_df, amcu_arr, affine):
                #find mean of aMCU raster within each building candidate in gdf (all bands)

                #buffer the polygon as edge pixels may not represent its 'real' aMCU values
                building_df.geometry = building_df.geometry.buffer(-1)

                gdf = building_df.copy()
                for band in range(7):
                    df_zonal_stats = pd.DataFrame(zonal_stats(building_df, amcu_arr[band],\
                                                              stats="mean", affine=affine,\
                                                              nodata=np.nan))
                    gdf = pd.concat([gdf, df_zonal_stats['mean']], axis=1).reset_index(drop=True)
                    gdf.rename(columns={'mean': f'band{band} mean'}, inplace=True)

                return gdf

            #KonaMauka and CCTrees probably don't have any (correctly-identified) buildings
            if 'KonaMauka' not in region and 'CCTrees' not in region:
                correct = poly_stats(correct, amcu_arr, affine)
                self.classified_candidates = pd.concat([self.classified_candidates, correct])

            incorrect = poly_stats(incorrect, amcu_arr, affine)
            self.classified_candidates = pd.concat([self.classified_candidates, incorrect])

        #mean=0.0 most likely indicates no data, don't want to use for training models
        self.classified_candidates = self.classified_candidates\
                                     [self.classified_candidates['band0 mean'] != 0.0]
        self.classified_candidates.reset_index(inplace=True, drop=True)


    def get_vars_from_df(self):
        """
        Extract feature and response variables from self.classified candidates. See
        self.classify_all_candidate_buildings and self.classify_with_rf_and_amcu.
        """

        #get X (feature) and y (response/target) variables
        #X contains mean aMCU in each band for each building candidate
        X = self.classified_candidates.drop(columns=['geometry', 'Values', 'Region'])
        size = self.classified_candidates.geometry.area
        X['size'] = size
        y = self.classified_candidates['Values']

        return X, y

    def create_reclassified_maps(self, model_dir, best_rf, X):
        """
        Write files containing maps in which building candidates are those identified
        as genuine by the RF model (see self.classify_with_rf_and_amcu)
        """

        #create a df that only contains building candidates retained by the RF model
        #done this way to avoid settingwithcopywarnings
        gdf = gpd.GeoDataFrame()
        gdf.loc[:, 'geometry'] = self.classified_candidates[['geometry']]
        gdf.loc[:, 'Region'] = self.classified_candidates[['Region']]

        #get the model predictions for all X
        gdf.loc[:, 'New Values'] = best_rf.predict(X)
        gdf = gdf[gdf['New Values'] == 1]

        #get the list of map files
        map_list = glob.glob(f'{model_dir}*cleaner*.shp')

        #iterate over the train+test regions/tiles/mosaics, writing the new buildings to file
        for map_file in map_list:

            #get the name of the test region
            region = map_file.split('votes_')[-1].replace('.shp', '')

            #create an output filename
            out_file = map_file.replace('cleaner', 'reclassified')

            #get the buildings for this region (if there are any)
            #undo buffering that was done in self.classify_all_candidate_buildings
            temp = gdf[gdf['Region'] == region].copy()
            temp.loc[:, 'geometry'] = temp.geometry.buffer(1)

            #write to shapefile
            if len(temp) > 0:
                temp.to_file(out_file, driver='ESRI Shapefile')

                #Rasterize and save (so we can compare 'before' and 'after' raster stats)
                #This raster is just a template
                with rasterio.open(map_file.replace('shp', 'tif'), 'r') as f:
                    arr = f.read()
                    meta = f.meta
                meta.update({'nodata': -9999.0}) #TODO: why is nodata not being propagated?
                self.rasterize_and_save(temp, arr, meta, out_file.replace('shp', 'tif'),\
                                        'New Values')
