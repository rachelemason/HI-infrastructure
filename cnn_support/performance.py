#!/usr/bin/env python3
#performance.py
#REM 2022-02-24


"""
Code for deriving mapping products from probability maps produced by
run_models.py/RunModels.ipynb, and creating various diagnostic plots.
"""

import os
from collections import defaultdict
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
import fiona
from shapely.geometry import shape, MultiPolygon, Polygon
from osgeo import gdal
import geopandas as gpd
from sklearn.metrics import classification_report


FEATURE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'
RESPONSE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_buildings/'
BOUNDARY_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_boundaries/'
GEO_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/geo_data/'
ROAD_FILES = [f'{GEO_PATH}SDOT_roads/{f}' for f in ['state_routes_SDOT_epsg32605.shp',\
                                                    'county_routes_SDOT_epsg32605.shp',\
                                                    'ServiceAndOtherRoads_SDOT_epsg32605.shp']]


class Utils():
    """
    Generic helper methods used by >1 class
    """

    def __init__(self, test_sets):
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
        ref_file = f'{RESPONSE_PATH}{self.test_sets[test_region]}_responses.tif'
        with rasterio.open(ref_file, 'r') as f:
            ref_arr = f.read()

        map_arr = map_arr[0]
        ref_arr = ref_arr[0]

        return map_arr, ref_arr


class MapManips():
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root, test_sets):
        self.model_output_root = model_output_root
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

            for word in ['model_', 'votes_']:
                if word in map_file:
                    test_region = map_file.split(word)[-1].replace('.tif', '')

            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()
                meta = f.meta

            #open the corresponding NDVI map
            ndvi_file = f'{FEATURE_PATH}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()

            #set pixels with NDVI > threshold to 0 i.e. not-building; write to file
            map_arr[ndvi_arr > ndvi_threshold] = 0
            outfile = map_file.replace('threshold', 'ndvi_cut')
            if verbose:
                print(f"Writing {outfile}")
                plt.imshow(map_arr[0])
                plt.show()
            with rasterio.open(f"{outfile}", 'w', **meta) as f:
                f.write(map_arr)


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


    def vectorize_and_clean(self, map_file, out_file, buffers, minpix=25):
        """
        Clean up coastal artefacts in an individual tile/test region map by vectorizing
        the map and rejecting polygons that don't lie wholly within (1) the coastline
        shapefile, and (2) the outline of the County TMK/Parcel map. Also, reject polygons
        that overlap with roads (in State DOT shapefiles). Then re-rasterize and save the
        'cleaned' file.
        """

        with rasterio.open(map_file) as f:
            arr = f.read()
            meta = f.meta

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

        #remove too-small polygons
        gdf = gdf.loc[gdf.geometry.area >= minpix]

        #Apply negative then positive buffer, to remove spiky bits and disconnect
        #polygons that are connected by a thread (they are usually separate buildings)
        gdf.geometry = gdf.geometry.buffer(buffers[0])
        #Delete polygons that get 'emptied' by negative buffering
        gdf = gdf[~gdf.geometry.is_empty]
        #Apply positive buffer to regain approx. original shape
        gdf.geometry = gdf.geometry.buffer(buffers[1])
        #If polygons have been split into multipolygons, explode into separate rows
        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        #Remove any interior holes (do after buffering)
        gdf.geometry = gdf.geometry.apply(lambda row: self._close_holes(row))

        #Remove polygons that intersect roads (do before getting any bounding rectangle)
        print('  - Removing polygons that intersect with roads')
        gdf = self._remove_roads(gdf)

        #Remove polygons that are outside the County parcel/TMK map outline, or outside the
        #coastline
        print('  - Removing polygons outside/overlapping the coast')
        gdf = self._remove_coastal_artefacts(gdf)

        #Rasterize and save what should be a 'cleaner' raster than the input one
        print(f'  - Rasterizing and saving {out_file}')
        meta.update({'compress': 'lzw'})
        with rasterio.open(out_file, 'w+', **meta) as f:
            shapes = ((geom,value) for geom, value in zip(gdf.geometry, gdf.Values))
            burned = rasterize(shapes=shapes, fill=0, out_shape=f.shape, transform=f.transform)
            burned[arr[0] < 0] = meta['nodata']
            f.write_band(1, burned)

        #Write the vector file as well
        gdf.to_file(out_file.replace('.tif', '.shp'), driver='ESRI Shapefile')


class Ensemble(MapManips):
    """
    Methods for creating ensemble maps
    """

    def __init__(self, model_output_root, test_sets, ensemble_path):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.ensemble_path = ensemble_path
        MapManips.__init__(self, model_output_root, test_sets)


    def create_ensemble(self, model_nums, region, map_type='threshold', show=True):
        """
        Create an ensemble map for a region, starting from a threshold,
        applied, ndvi_cut, or anmcu_cut map, by taking the sum of each map pixel.
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

        #write the ndvi_cut map to file
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

    def __init__(self, model_output_root, test_sets):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        Utils.__init__(self, test_sets)


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

            for word in ['model_', 'votes_']:
                if word in map_file:
                    test_region = map_file.split(word)[-1].replace('.tif', '')

            #open the map and response files
            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the NDVI file so we can get NDVI values for the candidates
            ndvi_file = f'{FEATURE_PATH}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
            ndvi_arr = ndvi_arr[0]#[20:-20, 20:-20]

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
        Produces a histogram of aMCU values for cleaned maps of test
        regions, for a single model run. Shows values for all pixels (sample thereof),
        and for pixels correctly and incorrectly classified as buildings.
        """

        print(f'Working on ndvi_cut maps from {model_dir}')

        all_amcu = defaultdict(list)
        correct_amcu = defaultdict(list)
        incorrect_amcu = defaultdict(list)

        #for each test_region map
        for map_file in glob.glob(f'{model_dir}/*cleaner*tif'):

            for word in ['model_', 'votes_']:
                if word in map_file:
                    test_region = map_file.split(word)[-1].replace('.tif', '')

            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the aMCU file, resampling to same resolution as map array
            amcu_file = f'{FEATURE_PATH}{self.test_sets[test_region]}_amcu.tif'
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

        for n, ax in enumerate(fig.axes):
            if n >= len(glob.glob(f'{model_dir}/*cleaner*tif')):
                ax.axis('off')
        plt.tight_layout()


    def amcu_scatter(self, model_dir, xband, yband, lims, incorrect):
        """
        Intended for producing scatters plot of two aMCU bands, normally 2 and 6.
        """

        fig, _ = plt.subplots(3, 3, figsize=(16, 8))

        #get the list of map files, move KonaMauka to the front so it can be used as a reference
        #the 3votes map for KonaMauka is hardwired, effort to fix that doesn't seem warranted
        map_list = glob.glob(f'{model_dir}/*cleaner*.tif')
        map_list.insert(0, map_list.pop(map_list.index\
                                        (f'{model_dir}cleaner_3votes_KonaMauka.tif')))

        #for each test_region map
        for map_file, ax in zip(map_list, fig.axes):

            #get the name of the test region
            test_region = map_file.split('votes_')[-1].replace('.tif', '')

            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #open the aMCU file, resampling to same resolution as map array
            amcu_file = f'{FEATURE_PATH}{self.test_sets[test_region]}_amcu.tif'
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
                    rng = np.random.default_rng(seed=0)
                    print('randomx')
                    kmx = rng.choice(kmx, 5000, axis=0).flatten().tolist()
                    kmx = [k for k in kmx if k > -9999]
                    print('randomy')
                    kmy = rng.choice(kmy, 5000, axis=0).flatten().tolist()
                    kmy = [k for k in kmy if k > -9999]
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


class Stats(Utils):
    """
    Methods for calculating and plotting performance statistics (map quality stats)
    """

    def __init__(self, model_output_root, test_sets, analysis_path):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.analysis_path = analysis_path
        Utils.__init__(self, test_sets)


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


    def raster_stats(self, model_dir, map_kind='ndvi_cut'):
        """
        Given a model class prediction array and a set of responses, calculate precision,
        recall, and f1-score for each class (currently assumes binary classes)
        """

        stats_dict = {}

        #get the list of map files
        map_list = glob.glob(f'{model_dir}/*{map_kind}*tif')

        #for each test_region map
        for map_file in map_list:

            if 'KonaMauka' in map_file:
                continue

            for word in ['model_', 'votes_']:
                if word in map_file:
                    test_region = map_file.split(word)[-1].replace('.tif', '')

            #get the map and labelled response files for this test region
            map_arr, ref_arr = self._get_map_and_ref_data(map_file, test_region)

            #get the boundary mask (requires response file as opposed to array)
            ref_file = f'{RESPONSE_PATH}{self.test_sets[test_region]}_responses.tif'
            boundary_file = f'{BOUNDARY_PATH}{self.test_sets[test_region]}_boundary.shp'

            #create an array of the same shape as the test region
            #in which everything outside the test dataset boundary/boundaries is NaN
            masko = self._boundary_shp_to_mask(boundary_file, ref_file)

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

        return stats_dict


    def stats_plot(self, stats_dict, plot_file):
        """
        Make a 9-panel plot in which each panel shows precision, recall, and f1-score
        for the test areas for a single model. At least as the models were orginally
        set up, rows=data type (DSM, eigenvalues, hillshade), columns=window size
        (16, 32, 64).
        """

        rows = int(np.ceil(len(stats_dict) / 3))
        fig, _ = plt.subplots(rows, 3, figsize=(12, 4*rows))
        for model, ax in zip(sorted(stats_dict.keys()), fig.axes):
            precision = []
            recall = []
            f1score = []
            regions = []
            for region, stats in sorted(stats_dict[model].items()):
                if region != 'KonaMauka':
                    precision.append((stats['1.0']['precision']))
                    recall.append((stats['1.0']['recall']))
                    f1score.append((stats['1.0']['f1-score']))
                    regions.append(region)
            x = range(len(precision))
            ax.plot(x, precision, color='b', ms=6, marker='o', label='precision')
            ax.plot(x, recall, color='r', ms=6, marker='o', label='recall')
            ax.plot(x, f1score, color='k', ms=6, marker='o', label='f1-score')
            ax.axhline(0.8, color='0.5', ls='--')
            ax.legend()
            ax.set_ylim(0, 1)
            ax.set_xticks(x, regions, rotation=45, ha='right')
            if 'ensemble' in plot_file:
                ax.set_title(f'Ensemble {model}votes')
            else:
                ax.set_title(f'Model #{model}')

        for n, ax in enumerate(fig.axes):
            if n >= len(stats_dict):
                ax.axis('off')

        plt.tight_layout()
        plt.savefig(self.analysis_path+plot_file, dpi=400)
            