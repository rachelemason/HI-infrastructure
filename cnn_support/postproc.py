#!/usr/bin/env python3
#postproc.py
#REM 2023-01-09

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
from sklearn.metrics import classification_report
from shapely.geometry import shape, MultiPolygon, Polygon
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
import fiona
import gdal
import geopandas as gpd

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


    @classmethod
    def _create_hist(cls, data, data_type, cut):
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

            #tiles 7, 12, 14, 15, 18, 19, 20, 21, 22, 31 can't/don't need to be cropped
            tile_crop_dict = {'tile008': [3000, -1, 0, -1], 'tile009': [14000, -1, 0, 18000],\
                              'tile016': [2500, -1, 0, 10000], 'tile024': [0, -1, 12500, -1],\
                              'tile025': [0, -1, 0, 20000], 'tile030': [0, -1, 15000, -1]}
            try:
                idx = tile_crop_dict[tile]
                nans = nans[:, idx[0]:idx[1], idx[2]:idx[3]]
                arr = arr[:, idx[0]:idx[1], idx[2]:idx[3]]
                plt.imshow(arr[0])
                plt.show()
                win = ((idx[0], idx[1]), (idx[2], idx[3]))
                profile = profile.copy()
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

        for tile in tiles:

            print(f'Working on {tile}')

            #get the (trimmed) map for this tile
            with rasterio.open(f'{self.out_path}trimmed_{n_votes}votes_{tile}.tif') as f:
                arr = f.read()
                meta = f.meta

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


class VectorManips():
    """
    Methods for vectorizing ensemble maps and improving building outlines
    """

    def __init__(self, data_path):
        self.data_path = data_path


    @classmethod
    def _close_holes(cls, geom):
        """
        Helper method for self.vectorise(). Removes interior holes from polygons.
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


    def _remove_roads(self, gdf, road_files):
        """
        Helper method for self.vectorise(). Reads roads from
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


    def vectorise(self, raster, minpix=25, road_files=None):
        """
        Vectorise a mosaic created by self.mosaic_and_crop(). Remove
        polygons containing fewer than <minpix> pixels, as they're probably
        false positives. Write a shapefile with the name as the input tif,
        <mosaic_name>.
        TODO: update docstring
        """

        with rasterio.open(f'{self.data_path}{raster}.tif', 'r') as f:
            arr = f.read()
            meta = f.meta

        print(f'Vectorizing {raster}')
        polygons = []
        values = []
        for vec in rasterio.features.shapes(arr.astype(np.int32), transform=meta['transform']):
            polygons.append(shape(vec[0]))
            values.append(vec[1])
        gdf = gpd.GeoDataFrame(crs=meta['crs'], geometry=polygons)

        gdf['Values'] = values

        #remove polygons composed of NaNs or 0s (building == 1)
        gdf = gdf[gdf['Values'] > 0]

        #remove too-small polygons
        gdf = gdf.loc[gdf.geometry.area >= minpix]

        #Remove any interior holes
        gdf.geometry = gdf.geometry.apply(lambda row: self._close_holes(row))

        #Apply negative then positive buffer, to remove spiky bits and disconnect
        #polygons that are connected by a thread (they are usually separate buildings)
        gdf.geometry = gdf.geometry.buffer(-3)
        #Delete polygons that get 'emptied' by negative buffering
        gdf = gdf[~gdf.geometry.is_empty]
        #Apply positive buffer to regain approx. original shape
        gdf.geometry = gdf.geometry.buffer(3)
        #If polygons have been split into multipolygons, explode into separate rows
        gdf = gdf.explode().reset_index(drop=True)

        #Get the oriented envelope/bounding rectangle
        boxes = gdf.apply(lambda row: row['geometry'].minimum_rotated_rectangle, axis=1)
        gdf.geometry = boxes

        #Remove polygons that intersect roads
        if road_files:
            gdf = self._remove_roads(gdf, road_files)

        print(f'There are {len(gdf)} building candidates in the {raster} map')

        file_name = raster+'.shp'
        print(f"Writing {self.data_path}{file_name}")
        gdf.to_file(f'{self.data_path}{file_name}TEMP', driver='ESRI Shapefile')


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
            with open(glob.glob(f'{model}*{model_kind}.json')[0], encoding='utf-8') as f:
                stats = json.load(f)
            for test_area in stats:
                for metric in ['precision', 'recall', 'f1-score']:
                    mean_stats[metric].append(test_area['1.0'][metric])
        for metric, values in mean_stats.items():
            print(metric, f'{np.mean(values):.2f}')


    def vector_stats(self, n_votes, vector_path, tolerance=1):
        """
        Calculate Intersection over Union for test region ensemble maps,
        then report IoU averaged over all.
        """

        for alias, name in self.apply_to.items():
            labels = gpd.read_file\
            (f"{DATA_PATH.replace('features', 'buildings')}{name}_responses.shp")
            result = gpd.read_file(f'{vector_path}ensemble_{n_votes}votes_{alias}.shp')

            l_coords = [(p.x, p.y) for p in labels.centroid]
            r_coords = [(p.x, p.y) for p in result.centroid]

            matches = {}
            #need to visualize results in QGIS and adjust tolerances
            for idx, coords in enumerate(l_coords):
                for idx2, coords2 in enumerate(r_coords):
                    if abs(coords[0] - coords2[0]) <= tolerance\
                    and abs(coords[1] - coords2[1]) <= tolerance:
                        matches[idx] = idx2

            matched_labels = labels.iloc[list(matches.keys())].reset_index(drop=True)
            matched_output = result.iloc[list(matches.values())].reset_index(drop=True)

            iou = {}
            for idx, row in matched_labels.iterrows():
                poly1 = row['geometry']
                poly2 = matched_output.loc[idx, 'geometry']
                intersect = poly1.intersection(poly2).area
                union = poly1.union(poly2).area
                iou[alias] = intersect / union
            print(f'{alias}, {np.mean(iou[alias]):.2f}')

            #Write out files for diagnostic purposes (check whether polygons matched well enough)
            matched_labels.to_file(f'{vector_path}{alias}_matched_labels', driver='ESRI Shapefile')
            matched_output.to_file(f'{vector_path}{alias}_matched_output', driver='ESRI Shapefile')

        print(f'Mean: {np.mean(list(iou.values())):.2f}')

        #Once polygon matches have been made, should be straightforward to rasterize
        #and calculate other stats as well

class Analysis():
    """
    Calculate various numbers needed for the mapping paper
    """

    def __init__(self, map_path, geo_path, analysis_path):
        self.map_path = map_path
        self.geo_path = geo_path
        self.analysis_path = analysis_path
        self.parcel_data = None #defined in parcel_ocean_distance()


    def total_area(self):
        """
        Find the area of the three geographies in sq. km by counting non-NaN pixels
        in maps (each pix is 1m sq).
        """

        count = 0
        for tif in ['SKona', 'NKona_SKohala', 'NHilo_Hamakua']:
            with rasterio.open(f'{self.map_path}{tif}.tif') as f:
                data = f.read()
            area = np.sum(data >= 0) * 1e-6
            print(f'Mapped area of {tif} is {area:.0f} sq. km')
            count += area

        print(f'Total mapped area is {count:.0f} sq. km')


    def count_buildings(self):
        """
        Count the number of buildings in our maps for the three geographies. For
        comparison with the number in the MSBFD (See count_msbfd)
        """

        count = 0
        for mapfile in ['SKona', 'NKona_SKohala', 'NHilo_Hamakua']:
            gdf = gpd.read_file(f'{self.map_path}{mapfile}/{mapfile}.shp')
            print(f'There are {len(gdf)} buildings in our {mapfile} map')
            count += len(gdf)
        print(f'We identified {count} buildings in total')


    def _create_mapped_region_shpfiles(self, region):
        """
        Helper method for count_msbfd and XX. Writes a shapefile that
        contains a single multipolygon representing the outline of one
        of our geographies, including the NaN (no-data) areas.
        """

        with rasterio.open(f'{self.map_path}{region}.tif', 'r') as f:
            arr = f.read()
            meta = f.meta

            #we want to find the overall map outline, so set all buildings (1) to not-building (0)
            arr[arr == 1] = 0

            #TODO: this should be a utility function, or inherited method
            print(f'Vectorizing {region}')
            polygons = []
            values = []
            for vec in rasterio.features.shapes(arr.astype(np.int32), transform=meta['transform']):
                polygons.append(shape(vec[0]))
                values.append(vec[1])
            mapped_region = gpd.GeoDataFrame(crs=meta['crs'], geometry=polygons)

            #create a multipolygon showing the mapped region outline
            mapped_region['Values'] = values
            mapped_region = mapped_region[mapped_region['Values'] >= 0]
            mapped_region = mapped_region.dissolve(by='Values').reset_index()

            mapped_region.to_file(f'{self.geo_path}{region}_mapped_region', driver='ESRI Shapefile')

            return mapped_region


    def count_msbfd(self):
        """
        Count the number of buildings in the MS Building Footprint Database that are
        within the non-NAN areas of the three DST geographies
        """

        #read the whole MS Building Footprint file for Hawai'i
        msbfd = gpd.read_file(f'{self.geo_path}Hawaii.geojson')
        msbfd = msbfd.to_crs(epsg=32605)

        count = 0
        for region in ['SKona', 'NKona_SKohala', 'NHilo_Hamakua']:

            if not os.path.exists(f'{self.geo_path}{region}_mapped_region'):
                mapped_region = self._create_mapped_region_shpfiles(region)
            else:
                mapped_region = gpd.read_file(f'{self.geo_path}{region}_mapped_region')

            if not os.path.exists(f'{self.geo_path}MSBFD_{region}'):
                #clip the MSBFD to the mapped region and count remaining buildings
                #save clipped MSBFD file with other shapefiles for future use
                print(f'Clipping {region}')
                msbfd_clip = gpd.clip(msbfd, mapped_region)
                msbfd_clip.to_file(f'{self.geo_path}MSBFD_{region}', driver='ESRI Shapefile')
            else:
                msbfd_clip = gpd.read_file(f'{self.geo_path}MSBFD_{region}')

            print(f'There are {len(msbfd_clip)} MS building footprints in {region}')
            count += len(msbfd_clip)

            #sanity check plot
            _, ax = plt.subplots(figsize=(12, 8))
            mapped_region.plot(color='cyan', ax=ax)
            msbfd_clip.plot(color='black', ax=ax)
            ax.set_axis_off()
            plt.show()

        print(f'There are {count} MS building footprints in our mapped areas')


    def parcels_buildings(self):
        """
        For each parcel in the Hawai'i County shapefile, find out whether the centroid of
        any of our mapped building rectangles falls within it. Create a dataframe containing
        parcel polygon, occupation status (True if contains any building centroids), and
        the area of the polygon in various units. Write results to shapefiles at self.analysis
        path, one for each of the three study areas. This is intended to be run only once, as
        it takes a couple of hours to get through everything.
        See geoffboeing.com/2016/10/r-tree-spatial-index-python/
        """

        #read the whole Hawai'i County parcel shapefile
        parcels = gpd.read_file(f'{self.geo_path}Parcels/Parcels_-_Hawaii_County.shp')
        parcels = parcels.to_crs(epsg=32605)

        #read the Hawai'i county Zoning shapefile
        zoning = gpd.read_file(f'{self.geo_path}Zoning/Zoning_(Hawaii_County).shp')
        zoning = zoning.to_crs(epsg=32605)

        #read the state coastline shapefile
        coast = gpd.read_file(f'{self.geo_path}Coastline/Coastline.shp')
        coast = coast[coast['isle'] == 'Hawaii'].reset_index(drop=True)
        coast = coast.to_crs(epsg=32605)

        self.parcel_data = {}
        for region in ['SKona', 'NKona_SKohala', 'NHilo_Hamakua']:

            matches = gpd.GeoDataFrame()

            #get the polygon defining the mapped region
            if not os.path.exists(f'{self.geo_path}{region}_mapped_region'):
                mapped_region = self._create_mapped_region_shpfiles(region)
            else:
                mapped_region = gpd.read_file(f'{self.geo_path}{region}_mapped_region')

            #get the parcel polygons, clipped to the mapped region
            if not os.path.exists(f'{self.geo_path}Parcels/Parcels_{region}'):
                print(f'Clipping {region}')
                parcels = gpd.clip(parcels, mapped_region)
                parcels.to_file(f'{self.geo_path}Parcels/Parcels_{region}',\
                                driver='ESRI Shapefile')
            else:
                parcels = gpd.read_file(f'{self.geo_path}Parcels/Parcels_{region}')

            print(f'There are {len(parcels)} parcels in {region}')
            parcel_polys = parcels.geometry.tolist()

            #read our building map for the same region
            buildings = gpd.read_file(f'{self.map_path}{region}/{region}.shp')
            print(f'There are {len(buildings)} buildings in {region}')

            #get building centroids, these points will be matched with parcels
            centroids = buildings.geometry.centroid

            #do the matching
            spatial_index = centroids.sindex
            for idx, parcel in enumerate(parcel_polys):
                #this will be the geometry column (has multipolygons so can't assign directly)
                matches.loc[idx, 'Parcel'] = parcel

                #find buildings whose centroids lie within the parcel
                possible_matches_index = list(spatial_index.intersection(parcel.bounds))
                possible_matches = centroids.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.intersects(parcel)]
                precise_matches = precise_matches.tolist()

                #0 buildings --> Not developed; >=1 building --> Developed
                #Encode as 0, 1 as can't save False, True in shapefiles
                if len(precise_matches) == 0:
                    matches.loc[idx, 'Occupied'] = '0'
                else:
                    matches.loc[idx, 'Occupied'] = '1'

            #convert parcel (multi)polygons into a proper geometry column
            matches['geometry'] = matches['Parcel']
            matches.drop(columns=['Parcel'], inplace=True)

            #add some area columns - different units for different purposes
            matches['m^2'] = matches.geometry.area
            matches['ft^2'] = matches.geometry.area * 10.7639
            matches['ha'] =  matches['m^2'] / 10000
            matches['acres'] =  matches['ft^2'] / 43560
            matches = matches.set_crs(epsg=32605)

            #record which parcels have buildings that are known to the county
            matches['County Bldg'] = parcels['bldgvalue'].map({0: '0'}).fillna('1')

            #now assign zoning to parcels
            spatial_index = zoning.sindex
            for idx, parcel in enumerate(parcel_polys):
                possible_matches_index = list(spatial_index.intersection(parcel.bounds))
                possible_matches = zoning.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.intersects(parcel)]

                if len(precise_matches) == 1:
                    matches.loc[idx, 'Zoning'] = precise_matches['zone'].tolist()[0]
                elif len(precise_matches) == 0:
                    matches.loc[idx, 'Zoning'] = 'Unknown'
                else:
                    #often there are several matching zones for a parcel. Sometimes that's real,
                    #but more often they seem to be spurious matches. I'm not sure what causes
                    #that but the spurious matches often seem to be neighbours of the 'real' one.
                    #Here we check the area of overlap for all matches, and only keep parcel-zone
                    #matches that overlap by >100 sq. m.
                    temp1 = []
                    for num, _ in precise_matches.reset_index().iterrows():
                        temp = gpd.GeoDataFrame({'geometry': [precise_matches.iloc[num].geometry],\
                                                 'zone': [precise_matches.iloc[num].zone]},\
                                                 crs='epsg:32605')
                        myparcel = gpd.GeoDataFrame({'geometry':[parcel]}, crs='epsg:32605')
                        overlay = gpd.overlay(temp, myparcel, how='intersection',\
                                              keep_geom_type=False)
                        if overlay.area[0] > 100:
                            temp1.append(temp['zone'].tolist()[0])
                    if len(temp1) == 1:
                        matches.loc[idx, 'Zoning'] = temp1[0]
                    elif len(temp1) >1:
                        temp1 = '_'.join(list(set(temp1)))
                        matches.loc[idx, 'Zoning'] = temp1
                    else:
                        matches.loc[idx, 'Zoning'] = 'Unknown'

            #calculate distance from ocean - TAKES A WHILE
            print('Calculating distance to coastline...')
            matches['Dist. (km)'] = [parcel.distance(coast.geometry[0].boundary) for parcel\
                                    in matches.geometry]
            matches['Dist. (km)'] = matches['Dist. (km)'] / 1000

            #put all this into class variable for further access
            self.parcel_data[region] = matches

            #also save to a shapefile for this region
            matches.to_file(f'{self.analysis_path}{region}_parcel_occupation.shp',\
                            driver='ESRI Shapefile')

            open_parcels = len(matches[matches['Occupied'] == '0'])
            print(f'There are approximately {open_parcels} undeveloped parcels in {region}')
            print('(Some parcels contain multiple buildings)')


    def parcel_properties(self, from_file=True):
        """
        Make a stacked histogram showing occupied and unoccupied parcels for
        each of the 3 geographies
        """

        _ = plt.figure()
        for n, region in enumerate(['SKona', 'NKona_SKohala', 'NHilo_Hamakua']):
            if from_file:
                data = gpd.read_file(f'{self.analysis_path}{region}_parcel_occupation.shp')
            else:
                data = self.parcel_data[region]
            occupied = data[data['Occupied'] == '1']
            empty = data[data['Occupied'] == '0']
            plt.bar(n, len(occupied), color='0.5', edgecolor='k',\
                    label='Occupied' if n == 0 else '')
            plt.bar(n, len(empty), bottom=len(occupied), color='w', edgecolor='k',\
                   label='Unoccupied' if n == 0 else '')
        plt.legend()
        plt.xticks([0, 1, 2], ['SKona', 'NKona_SKohala', 'NHilo_Hamakua'])
        plt.ylabel('Number of parcels')


    @classmethod
    def _column_label(cls, col):
        """
        Helper method for self.parcel_plot. For a given column name, return the corresponding
        x axis label
        """

        col_dict = {'ha': 'Area, ha', 'Dist. (km)': 'Distance from coastline, km'}
        return col_dict[col]


    def parcel_plot(self, column, from_file=True, binwidth=200, xmax=5, ymax=1000):
        """
        Make a multi-panel set of histograms showing n as a function of <column>,
        where <column> is a column in the parcel occupation file. For example,
        show number of parcels as a function of distance from the coast. One panel
        per geography.
        """

        fig, _ = plt.subplots(2, 2, figsize=(16, 8))

        for region, ax in zip(['SKona', 'NKona_SKohala', 'NHilo_Hamakua'], fig.axes):

            if from_file:
                data = gpd.read_file(f'{self.analysis_path}{region}_parcel_occupation.shp')
            else:
                data = self.parcel_data[region]
            occupied = data[data['Occupied'] == '1']
            empty = data[data['Occupied'] == '0']

            bins = np.arange(0, xmax, binwidth)
            ax.hist([occupied[column], empty[column]], bins=bins, stacked=True,\
                    histtype='stepfilled', color=['0.5', 'white'], edgecolor='k',\
                    label=['Occupied', 'Unoccupied'])

            ax.legend()
            ax.set_ylim(0, ymax)
            ax.set_xlabel(self._column_label(column))
            ax.set_ylabel('Number of parcels')
            ax.set_title(f'{region}')

        for n, ax in enumerate(fig.axes):
            if n == 3:
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(f'{self.analysis_path}figures/parcels_by_coastal_distance.png', dpi=400)
