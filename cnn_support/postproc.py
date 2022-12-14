#!/usr/bin/env python3
#postproc.py
#REM 2022-12-12

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

import os
import random
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from skimage import measure
from sklearn.metrics import classification_report
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.mask import mask
import fiona
import gdal
from shapely.geometry import shape, asShape

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
            plt.axvline(cut[0])
            plt.axvline(cut[1])
        plt.legend(loc='upper left')
        plt.xlabel(data_type)
        plt.ylabel('Ill-defined frequency-related y-label...')

        #NOW SAVE THE FIGURE


    def histo(self, data_type='ndvi', band=0, threshold=0.85, cut_range=None):
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

        if cut_range is not None:
            print(f'There are {len(building_pix)} candidate building pixels')
            num = len([b for b in building_pix if (b > cut_range[0] and b < cut_range[1])])
            print(f'Applying the cut preserves {num} pixels')

        self._create_hist([all_pix, building_pix], data_type, cut_range)


class Ensemble():
    """
    Do ensemble averaging of maps
    """

    def __init__(self, ensemble_path, out_path, apply_to, model_set):
        self.ensemble_path = ensemble_path
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

            #get the relevant maps, for each test region
            for model_dir in self.model_set:
                this_file = glob.glob(f'{model_dir}{start_from}_{region_name}.tif')[0]
                this_array, meta = open_raster(this_file)
                model_list.append(this_array)


            if start_from == 'applied_model':
                #take the MEAN of the probabilities
                combo = np.mean(model_list, axis=0)
                #do this in order to have building class=1 when starting from prob. maps
                combo = 1 - combo

                #apply a threshold to convert to binary classes
                classes = np.zeros((combo.shape[0], combo.shape[1]))
                classes[combo == -9999] = -9999
                classes[np.logical_and(combo >= threshold, classes != -9999)] = 1

                #apply the ndvi cut
                ndvi_file = glob.glob(f'{DATA_PATH}{region_data}_ndvi_hires.tif')[0]
                with rasterio.open(ndvi_file, 'r') as f:
                    ndvi = f.read()
                cut_classes = np.expand_dims(classes, 0).astype(np.float32)
                cut_classes[ndvi > ndvi_threshold] = 0
            else:
                #find the SUM of the maps, which can be interpreted as the number of 'votes'
                #for a candidate building pixel
                combo = np.sum(model_list, axis=0)
                cut_classes = np.expand_dims(combo, 0).astype(np.float32)

            if show:
                plt.imshow(cut_classes[0][20:-20, 20:-20])
                plt.show()

            #write the ndvi_cut map to file
            os.makedirs(self.out_path, exist_ok=True)
            meta.update(count=1)
            with rasterio.open(f"{self.ensemble_path}ndvi_cut_model_{region_name}.tif", 'w',\
                               **meta) as f:
                f.write(cut_classes)

            ensemble[region_name] = cut_classes[0]

        return ensemble


    def choose_cut_level(self, ensemble, name, n_votes):
        """
        For an ensemble created by self.create_ensemble() with <start_from>='ndvi_cut_model',
        return a version converted to binary classes using <n_votes>. The input ensemble will have
        pixel values = 0-n in steps of 1, where n is the number of models that went into the
        ensemble. For example, an ensemble map created from 5 individual models has pixel values
        like this:
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
        
        input_map = f"{self.ensemble_path}ndvi_cut_model_{name}.tif"
        print(f'Creating {self.out_path}ensemble_{n_votes}votes_{name}.tif')
        
        if ensemble == 'from_file':
            arr, meta = open_raster(input_map)
        else:
            arr = ensemble.copy()
            with rasterio.open(input_map, 'r') as f:
                meta = f.meta
                
        arr[arr < n_votes] = 0
        arr[arr >= n_votes] = 1

        #write the map to file
        binary_map = np.expand_dims(arr, 0).astype(np.float32)
        with rasterio.open(f"{self.out_path}ensemble_{n_votes}votes_{name}.tif", 'w',\
                            **meta) as f:
            f.write(binary_map)

        return arr

    
    def trim_ensemble_tiles(self, tile_list, n_votes):
        """
        Trim ensemble tiles and LiDAR data to more manageable sizes, by removing
        areas that are all NaN.
        """
        
        nanpath = '/data/gdcsdata/HawaiiMapping/Full_Backfilled_Tiles/'
        
        for tile in tile_list:
            #Get the original LiDAR for this tile, which tells us where no-data regions are
            print(f'Filling in NaN values for {tile} from DSM')
            with rasterio.open(f'{nanpath}{tile}/{tile}_backfilled_surface_1mres.tif') as f:
                nans = f.read()
                nan_meta = f.meta
                
            #get the map for this tile
            with rasterio.open(f'{self.out_path}ensemble_{n_votes}votes_{tile}.tif') as f:
                arr = f.read()
                profile = f.profile
            
            arr[nans == nan_meta['nodata']] = -9999

            plt.imshow(arr[0], vmin=-9999, vmax=1)
            plt.show()

            #tiles 7, 12, 14, 15, 19, 20, 21, 22, 31 can't/don't need to be cropped
            tile_crop_dict = {'tile008': [3000, -1, 0, -1], 'tile009': [14000, -1, 0, 18000],\
                              'tile016': [2500, -1, 0, 10000], 'tile018': [],\
                              'tile024': [0, -1, 12500, -1], 'tile025': [0, -1, 0, 20000],\
                              'tile030': [0, -1, 15000, -1]}
            try:
                idx = tile_crop_dict[tile]
                nans = nans[:, idx[0]:idx[1], idx[2]:idx[3]]
                arr = arr[:, idx[0]:idx[1], idx[2]:idx[3]]
                plt.imshow(arr[0])
                plt.show()
                win=((idx[0], idx[1]),(idx[2], idx[3]))
                profile=profile.copy()
                profile['width'] = arr.shape[2]
                profile['height'] = arr.shape[1]
                profile['transform'] = f.window_transform(win)
            except KeyError:
                pass

            #TODO: turn this into symbolic links for tiles that didn't need trimming
            print(f'Writing {tile} file')
            with rasterio.open(f'{self.out_path}trimmed_{n_votes}votes_{tile}.tif',\
                               'w', **profile) as f:
                f.write(arr)

    
    def nans_and_edges(self, tiles, n_votes, edge_buf=30):
        """
        Add NaNs to the ensemble for this tile (specified by <tile> and <n_votes> where
        there are no data (as determined by the DSM for <tile>), and set a buffer around
        edge regions to NaN. Writes an 'edges_cleaned' file for the ensemble for this tile.
        This needs a LOT of memory to run on full tiles, especially those with a lot of NaN
        pixels. 80Gb seems to work in all cases except tile009, which has to be cropped.
        """
        
        #nanpath = '/data/gdcsdata/HawaiiMapping/Full_Backfilled_Tiles/'

        for tile in tiles:
            
            print(f'Working on {tile}')
                        
            #Get the original LiDAR for this tile, which tells us where no-data regions are
            #print('Filling in NaN values from DSM')
            #with rasterio.open(f'{nanpath}{tile}/{tile}_backfilled_surface_1mres.tif') as f:
            #    nans = f.read()
            #    nan_meta = f.meta
                
            #get the (trimmed) map for this tile
            with rasterio.open(f'{self.out_path}trimmed_{n_votes}votes_{tile}.tif') as f:
                arr = f.read()
                meta = f.meta
                
            #set the relevant parts of the map to -9999 (where there were no LiDAR data)
            #arr[nans == nan_meta['nodata']] = -9999

            #plt.imshow(arr[0], vmin=-9999, vmax=1)
            #plt.show()
            
            #Now, work on extending the NaN regions to 'rub out' the edge effects

            #get the indexes of the NaN (-9999) values in the map
            print('...getting all NaN indices')
            idx1 = np.where(arr[0] < 0)
            idx = list(zip(idx1[0], idx1[1]))

            #find the indices of the NaNs that don't just have NaNs as neighbours -
            #these mark the edges we want to edit around
            print('...getting edge indices')
            idx2 = []
            for i in idx:
                neighbors = arr[:, i[0]-1:i[0]+1+1, i[1]-1:i[1]+1+1]
                if len(np.unique(neighbors)) > 1:
                    idx2.append(i)

            #set the pixels within <edge_buffer> of these indices to -9999
            print('...editing edge indices')
            for i in idx2:
                arr[:, i[0]-edge_buf:i[0]+edge_buf+1, i[1]-edge_buf:i[1]+edge_buf+1] = -9999
                
            #plot and save the edited map tile
            plt.imshow(arr[0], vmin=-9999, vmax=1)
            
            print('...writing file')
            with rasterio.open(f'{self.out_path}edges_cleaned_{tile}.tif', 'w', **meta) as temp:
                temp.write(arr)

    
    def mosaic_and_crop(self, tiles, shapefile, outname):
        """
        Mosaic a set of 'edges_cleaned' tiles and crop to the boundaries of
        <shapefile>
        """

        print(f'Mosaicking {outname} tiles')
        to_mosaic = []
        for tile in tiles:
            raster = rasterio.open(f'{self.out_path}edges_cleaned_{tile}.tif')
            to_mosaic.append(raster)
            meta = raster.meta

        mosaic, out_trans = merge(to_mosaic)

        meta.update({"height": mosaic.shape[1], "width": mosaic.shape[2], "transform": out_trans})
        with rasterio.open(f'{self.out_path}temp.tif', "w", **meta) as f:
            f.write(mosaic)

        print(f'Cropping to {outname} boundaries')
        with fiona.open(shapefile, "r") as shpf:
            shapes = [feature["geometry"] for feature in shpf]

        with rasterio.open(f"{self.out_path}temp.tif") as f:
            cropped, out_trans = rasterio.mask.mask(f, shapes, crop=True)
            meta = f.meta
            
        meta.update({"height": cropped.shape[1], "width": cropped.shape[2], "transform": out_trans})
        with rasterio.open(f'{self.out_path}{outname}.tif', "w", **meta) as f:
            f.write(cropped)

        os.remove(f"{self.out_path}temp.tif")
        

    def vectorise(self, mosaic_name, minpix=25):
        """
        Vectorise a mosaic created by self.mosaic_and_crop(). Remove
        polygons containing fewer than <minpix> pixels, as they're probably
        false positives. Write a shapefile with the name as the input tif,
        <mosaic_name>.
        """
        
        with rasterio.open(f'{self.out_path}{mosaic_name}.tif', 'r') as f:
            arr = f.read()
            meta = f.meta
            
        print(f'Vectorizing {mosaic_name}')
        polygons = []
        values = []
        for vec in rasterio.features.shapes(arr.astype(np.int32), transform=meta['transform']):
            polygons.append(shape(vec[0]))
            values.append(vec[1])
        gdf = gpd.GeoDataFrame(crs=meta['crs'], geometry=polygons)
        
        gdf['Values'] = values
        
        #remove very small polygons
        gdf = gdf.loc[gdf.geometry.area >= minpix]
        
        #remove polygons composed of NaNs
        gdf = gdf[gdf['Values'] > 0]
        
        #last polygon covers entire array, so delete it
        #TODO: is this now necessary?
        #gdf = gdf.head(-1)

        mosaic_name = mosaic_name.replace('tif', 'shp')
        print(f"Writing {self.out_path}{mosaic_name}")
        gdf.to_file(f'{self.out_path}{mosaic_name}', driver='ESRI Shapefile')


class Performance():
    """
    Methods for assessing model performance/map quality
    """

    def __init__(self, apply_to, model_set):
        self.apply_to = apply_to
        self.model_set = model_set


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


    def get_individual_models(self, model_type='ndvi_cut'):
        """
        Return arrays for the models (maps) that were used to create the ensembles,
        so their performance can be compared
        """

        super_dict = {}
        for idx, model_dir in enumerate(self.model_set):
            model_dict = {}
            for region_name in self.apply_to.keys():
                this_file = glob.glob(f'{model_dir}{model_type}_model_{region_name}.tif')[0]
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


    @classmethod
    def load_stats(cls, model_set, model_kind='threshold'):
        """
        Print precision, recall, and f1-score averaged over all test areas in
        all <model_kind> models used in the ensembles. This gives the stats in
        rows 1 and 2 in Table 1 of the infrastructure paper.
        """
        
        print(f'Stats averaged over all test areas, all {model_kind} models in the ensembles:')
        
        mean_stats = {'precision': [], 'recall': [], 'f1-score': []}
        for model in model_set:
            with open (glob.glob(f'{model}*{model_kind}.json')[0]) as f:
                stats = json.load(f)
            for test_area in stats:
                for metric in ['precision', 'recall', 'f1-score']:
                    mean_stats[metric].append(test_area['1.0'][metric])
        for metric, values in mean_stats.items():
            print(metric, f'{np.mean(values):.2f}')
            