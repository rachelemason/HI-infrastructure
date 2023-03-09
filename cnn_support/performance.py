#!/usr/bin/env python3
#performance.py
#REM 2022-03-09


"""
Code for deriving mapping products from probability maps produced by
run_models.py/RunModels.ipynb, and creating various diagnostic plots.
"""

import os
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterstats import zonal_stats
import fiona
from shapely.geometry import shape, MultiPolygon, Polygon
from osgeo import gdal
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,\
jaccard_score
from scipy.stats import randint, hmean
import seaborn as sns

pd.set_option('display.precision', 2)

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

    def __init__(self, all_labelled_data):
        self.all_labelled_data = all_labelled_data


    def _get_map_and_ref_data(self, map_file, region):
        """
        Helper method that just reads the map and labelled response
        data for the specified region, clips the edges, and returns
        both as 2D arrays
        """

        #open the map
        with rasterio.open(map_file, 'r') as f:
            map_arr = f.read()

        #open the corresponding reference (labelled buildings) file
        ref_file = f'{RESPONSE_PATH}{self.all_labelled_data[region]}_responses.tif'
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
        temp = [idx for idx, _ in enumerate(r_coords) if idx not in matches.values()]
        unmatched_candx = result.iloc[temp].reset_index(drop=True)
        unmatched_candx['Values'] = 0
        
        #find all the labeled buildings that were not detected
        #false negatives
        temp = [idx for idx, _ in enumerate(l_coords) if idx not in matches]
        unmatched_labels = labels.iloc[temp].reset_index(drop=True)
        unmatched_labels['Values'] = 0

        return matched_candx, unmatched_candx, matched_labels, unmatched_labels


    @classmethod
    def rasterize_and_save(cls, gdf, arr, meta, out_file, column):
        """
        Given a geodataframe and raference array/metadata, rasterize
        the file using <column> values and save to <out_file>.
        """

        print(f'  - Rasterizing and saving {out_file}')
        meta.update({'compress': 'lzw'})
        with rasterio.open(out_file, 'w+', **meta) as f:
            shapes = ((geom,value) for geom, value in zip(gdf['geometry'], gdf[column]))
            burned = rasterize(shapes=shapes, fill=0, out_shape=f.shape, transform=f.transform)
            burned[arr[0] < 0] = meta['nodata']
            f.write_band(1, burned)


    def remove_small_buildings(self, minsize=50):
        """
        Create new version of labelled response files with buildings with area
        <= <minsize> removed. Then, rasterize those files. This doesn't belong
        at all in this module/workflow, but I've resorted to putting it here for
        reasons to do with environments and packages...
        """

        #directory initially containing copies of shapefiles in labeled_region_buildings dir
        path = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_buildings2/*shp'
        files = glob.glob(path)

        for f in files:

            gdf = gpd.read_file(f)
            gdf = gdf.set_crs(epsg=32605)
            gdf = gdf.loc[gdf.geometry.area > minsize]

            if len(gdf) > 0:
                gdf.loc[:, 'FID'] = 1
                gdf.to_file(f, driver='ESRI Shapefile')

                template = f.replace('shp', 'tif').replace('buildings2', 'buildings')
                with rasterio.open(template, 'r') as src:
                    arr = src.read()
                    meta = src.meta
                meta.update({'nodata': -9999.0}) #TODO: why is nodata not being propagated?
                self.rasterize_and_save(gdf, arr, meta, f.replace('shp', 'tif'), column='FID')
            else:
                print(f'****Not handling {f}, copy it over yourself****')


class MapManips(Utils):
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root, test_sets, all_labelled_data):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.all_labelled_data = all_labelled_data
        self.classified_candidates = None #(defined in classify_all_candidate_buildings)
        Utils.__init__(self, all_labelled_data)


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

            region = [reg for reg in self.all_labelled_data if reg in map_file][0]

            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()
                meta = f.meta

            #open the corresponding NDVI map
            ndvi_file = f'{FEATURE_PATH}{self.all_labelled_data[region]}_ndvi_hires.tif'
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
        self.rasterize_and_save(gdf, arr, meta, out_file, 'Values')

        #Write the vector file as well
        gdf.to_file(out_file.replace('.tif', '.shp'), driver='ESRI Shapefile')


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


    def get_Xy_from_df(self):
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


    def classify_with_rf_and_amcu(self, X, y, show_confusion_matrix=True,\
                                  show_feature_importance=True, n_iter=50):
        """
        Fit random forest models to the building candidates in self.classified candidates, which
        are labelled 0 or 1 depending on whether they are false positives or genuine buildings.
        Tune the n_estimators and max_depth hyperparameters, and use the best-performing model
        to reclassify all the buildings according to whether the model thinks they are real or not.
        Creates self.reclassified, a df that can be written back to shapefiles (cleaned maps) for
        individual regions using self. create_reclassified_maps.
        See www.datacamp.com/tutorial/random-forests-classifier-python
        """

        #split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        #create a random forest classifier
        #use random search to find the best hyperparameters
        rand_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=\
                                         {'n_estimators': randint(50, 500),\
                                          'max_depth': randint(1, 20),\
                                          'min_samples_split' : randint(2, 10),\
                                          'min_samples_leaf' : randint(1, 5)}, n_iter=n_iter, cv=5,\
                                          n_jobs=-1)

        #fit the random search object to the data
        rand_search.fit(X_train, y_train)

        #get the best-performing model
        best_rf = rand_search.best_estimator_

        #print the best hyperparameters
        print('Best accuracy:',  rand_search.best_score_)
        print('Best hyperparameters:',  rand_search.best_params_)

        #predict classes for test data
        y_pred = best_rf.predict(X_test)

        #show the classification report
        print(classification_report(y_test, y_pred))

        #show the confusion matrix
        if show_confusion_matrix:
            _ = plt.figure()
            cmat = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cmat).plot()

        #show the feature importances
        if show_feature_importance:
            _ = plt.figure()
            feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).\
                             sort_values(ascending=False)
            feature_importances.plot.bar()

        return best_rf


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


class Ensemble(MapManips):
    """
    Methods for creating ensemble maps
    """

    def __init__(self, model_output_root, test_sets, ensemble_path, all_labelled_data):
        self.model_output_root = model_output_root
        self.test_sets = test_sets
        self.ensemble_path = ensemble_path
        MapManips.__init__(self, model_output_root, test_sets, all_labelled_data)


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

    def __init__(self, model_output_root, all_labelled_data):
        self.model_output_root = model_output_root
        self.all_labelled_data = all_labelled_data
        Utils.__init__(self, all_labelled_data)


    @classmethod
    def _histo(cls, ax, data, bins, xtext, xlims, xlinepos=None, legend=True):
        """
        Helper method for self.ndvi_hist. Puts the data into
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
        Produces a histogram of NDVI values for threshold maps of training+test
        regions, for a single model run. Shows values for all pixels (sample thereof),
        and for pixels correctly and incorrectly classified as buildings.
        """

        print(f'Working on threshold maps from {model_dir}')

        all_ndvi = []
        correct_ndvi = []
        incorrect_ndvi = []

        #for each test_region map
        for map_file in glob.glob(f'{model_dir}/*threshold*'):

            region = [reg for reg in self.all_labelled_data if reg in map_file][0]

            #open the map and response files
            map_arr, ref_arr = self._get_map_and_ref_data(map_file, region)

            #open the NDVI file so we can get NDVI values for the candidates
            ndvi_file = f'{FEATURE_PATH}{self.all_labelled_data[region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
            ndvi_arr = ndvi_arr[0]

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
        
            #is this a test region map? Or a training region map?
            res = [region for region in self.test_sets.keys() if region in map_file]

            #if this isn't a test region map, we'll exit and not calculate stats for it
            if len(res) == 0:
                continue

            test_region = res[0]

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
            stats_dict[test_region].update({'jaccard': jaccard_score(expected, predicted)})

        return stats_dict


    @classmethod
    def display_raster_stats(cls, stats_dict):
        """
        Display a data frame containing the stats obtained by self.raster_stats.
        """
        
        df = pd.DataFrame()
        for item in stats_dict:
            df.loc[item, 'Precision'] = stats_dict[item]['1.0']['precision']
            df.loc[item, 'Recall'] = stats_dict[item]['1.0']['recall']
            df.loc[item, 'F1-score'] = stats_dict[item]['1.0']['f1-score']
            df.loc[item, 'IoU'] = stats_dict[item]['jaccard']
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
        for model, ax in zip(sorted(stats_dict.keys()), fig.axes):
            precision = []
            recall = []
            f1score = []
            regions = []
            iou = []
            for region, stats in sorted(stats_dict[model].items()):
                if region != 'KonaMauka':
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
        
        
    def vector_stats(self, model_dir, map_kind='reclassified'):
        """
        Given a shapefile of building candidates and a labelled response file,
        calculate precision, recall, f1-score, and IOU
        """

        stats_df = pd.DataFrame()
        self.matched_candx = pd.DataFrame()
        self.matched_labels = pd.DataFrame()
        self.unmatched_labels = pd.DataFrame()
        self.unmatched_candx = pd.DataFrame()

        #get the list of map files
        map_list = glob.glob(f'{model_dir}*{map_kind}*shp')

        #for each test_region map
        for map_file in map_list:

            #is this a test region map? Or a training region map?
            res = [region for region in self.test_sets.keys() if region in map_file]

            #if this isn't a test region map, we'll exit and not calculate stats for it
            if len(res) == 0:
                continue

            test_region = res[0]

            labels = gpd.read_file(f"{RESPONSE_PATH}{self.test_sets[test_region]}_responses.shp")
            result = gpd.read_file(f'{model_dir}{map_kind}_{test_region}.shp')
            
            matched_candx, unmatched_candx, matched_labels, unmatched_labels = \
                                    self.label_building_candidates(labels=labels, result=result)
            self.matched_labels = pd.concat([self.matched_labels, matched_labels])
            self.matched_candx = pd.concat([self.matched_candx, matched_candx])
            self.unmatched_labels = pd.concat([self.unmatched_labels, unmatched_labels])
            self.unmatched_candx = pd.concat([self.unmatched_candx, unmatched_candx])

            #Calculate precision, recall and f1-score for each test region
            #precision = TP/(TP+FP)
            #recall = TP/(TP+FN)
            tempdict = {}
            tempdict['precision'] = len(matched_candx)/(len(matched_candx)\
                                                             +len(unmatched_candx))
            tempdict['recall'] = len(matched_candx)/(len(matched_candx)\
                                                          +len(unmatched_labels))
            tempdict['f1-score'] = hmean([tempdict['precision'], tempdict['recall']])

            #Calculate IoU for each test region
            iou = []
            for idx, row in matched_labels.iterrows():
                poly1 = row['geometry']
                poly2 = matched_candx.loc[idx, 'geometry']
                intersect = poly1.intersection(poly2).area
                union = poly1.union(poly2).area
                iou.append(intersect / union)
            tempdict['iou'] = np.mean(iou)
            print('***PROBABLY NEED TO EDIT IOU CALC TO INCLUDE FALSE POSITIVES/NEGATIVES***')

            for name, stat in zip(list(tempdict.keys()), list(tempdict.values())):
                stats_df.loc[test_region, name] = stat

            #Write out files for diagnostic purposes
            for gdf, name in zip ([matched_labels, matched_candx, unmatched_labels,\
                                   unmatched_candx], ['matched_labels', 'matched_candidates',\
                                   'unmatched_labels', 'unmatched_candidates']):
                if len(gdf) > 0:
                    #Sometimes there are no unmatched X so no point saving
                    gdf.to_file(f'{model_dir}{test_region}_{name}.shp', driver='ESRI Shapefile')
                    
        stats_df.loc['Mean'] = stats_df.mean()
        display(stats_df)

        
    def building_size_plots(self):
        """
        Create a multi-part figure showing (1) detection rate vs building size, and
        (2) mapped size vs manual size
        """
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        print('NB: AX1 X axis limits exclude a small number of very large buildings (>1000 sq. m)')
        data = pd.concat([self.matched_labels, self.unmatched_labels])
        bins = np.arange(0, 1020, 10)
        ax1.hist(data.geometry.area, bins=bins, label='All manually-labeled buildings')
        ax1.hist(self.matched_labels.geometry.area, bins=bins, color='limegreen', histtype='step',\
                 lw=2, label='Detected buildings')
        ax1.hist(self.unmatched_labels.geometry.area, bins=bins, color='gold', histtype='step',\
                 lw=2, label='Non-detections')
        ax1.set_xlim(0, 1020)
        ax1.set_ylim(0, 65)
        ax1.legend()
        ax1.set_xlabel('Manually-outlined building size, sq. m')
        ax1.set_ylabel('Number of buildings')
        
        print('NB: AX2 Y axis limits may exclude outliers (if vectorise.bbox=True)')
        data = pd.DataFrame()
        data['Manual size'] = self.matched_labels.geometry.area
        data['Model size'] = self.matched_candx.geometry.area
        sns.scatterplot(data=data, x='Manual size', y='Model size', ax=ax2)
        sns.kdeplot(data=data, x='Manual size', y='Model size', fill=True, cmap='viridis',\
                    alpha=0.5, ax=ax2)
        sns.regplot(data=data, x='Manual size', y='Model size', scatter=False, color='r')
        ax2.set_xlim(0, 1000)
        ax2.set_ylim(0, 1000)
        ax2.set_xlabel('Manually-outlined building size, sq. m')
        ax2.set_ylabel('Modeled building size, sq. m')
        ax2.plot([0, 1000], [0, 1000], ls='--', color='k')


class MLClassify(MapManips):
    """
    Methods for classifying buildings based on applying random forest model
    to aMCU data. This didn't work well so this code will probably be removed,
    just moving it here now to keep things tidy just in case it's needed later.
    """

    def __init__(self, model_output_root, test_sets, all_labelled_data):
        self.all_labelled_data = all_labelled_data
        MapManips.__init__(self, model_output_root, test_sets, all_labelled_data)

    def get_Xy_from_rasters(self):
        """
        Reads in all labelled building raster files, gets aMCU values for each pixel,
        reformats as X, y variables for RF fitting with self.classify_with_rf_and_amcu.
        """

        y = []
        X = [[] for x in range(7)]

        #for each reference region
        for region, data in self.all_labelled_data.items():
            print(region)

            #open the response files - this is where the labels are from
            with rasterio.open(f'{RESPONSE_PATH}{data}_responses.tif') as f:
                ref_arr = f.read()

            #open the aMCU file for the test region, which will give our X variable
            amcu_file = f'{FEATURE_PATH}{data}_amcu_hires.tif'
            with rasterio.open(amcu_file, 'r') as f:
                amcu_arr = f.read()

            #reshape 2D to 1D so we can remove NaNs
            ref_arr = np.ravel(ref_arr)
            amcu_arr = np.reshape(amcu_arr, (amcu_arr.shape[0], -1))

            #now remove the NaNs. Some considerations/complications:
            # -- there aren't any Nans in ref_arr because these regions were selected to
            #    not go off the edges of the LiDAR etc.
            # -- NaN in the aMCU maps actually seems to be 0.0, not the usual -9999.0
            # -- Also, different aMCU bands appear to have (slightly) different numbers of 0.0s,
            #    with band1 seeming to have the most
            #There must be a more efficient way of doing this, but...

            for band, data in enumerate(amcu_arr):
                X[band].extend([val for idx, val in enumerate(data) if amcu_arr[1][idx] != 0])
            y.extend(ref_arr[amcu_arr[1] != 0])

        #reformat as needed by sklearn for RF fitting
        X = np.array(X).T
        y = np.array(y)

        np.save(f'{RESPONSE_PATH}X_for_fitting', X)
        np.save(f'{RESPONSE_PATH}y_for_fitting', y)

        return X, y


    @classmethod
    def sample_from_Xy(cls, X, y, sample=10000):
        """
        Divide large X and y arrays into arrays including only elements
        where y=0 or y=1, then take a random sample from each of those. Doing it
        that way means that buildings are better represented than they would be in
        a simple random sample from all elements. Sampling is needed because the
        original arrays are too large to fit modesl to.
        """

        def sample_zeros_or_ones(value):
            temp_y = y[y==value] #labels where building=0
            temp_x = X[y==value] #amcu values for pixels where building=0
            idx = np.random.choice(np.arange(temp_y.shape[0]), sample, replace=False)
            temp_y = temp_y[idx]
            temp_x = temp_x[idx]

            return temp_x, temp_y

        zeros_x, zeros_y = sample_zeros_or_ones(value=0)
        ones_x, ones_y = sample_zeros_or_ones(value=1)
        y = np.concatenate((ones_y, zeros_y), axis=0)
        X = np.concatenate((ones_x, zeros_x), axis=0)

        return X, y


    def create_rf_classified_rasters(self, best_rf, outpath):
        """
        Use the best rf model to predict building classes based on aMCU and
        save the result to a tif file in <outpath>.
        """

        #for each reference region
        for region, data in self.all_labelled_data.items():
            print(region)

            #open the aMCU file for the test region
            amcu_file = f'{FEATURE_PATH}{data}_amcu_hires.tif'
            with rasterio.open(amcu_file, 'r') as f:
                amcu_arr = f.read()
                meta = f.meta

            #reshape aMCU as necessary, get predicted classes, re-reshape
            original_shape = amcu_arr[0].shape
            amcu_arr = np.reshape(amcu_arr, (amcu_arr.shape[0], -1)).T
            predicted = best_rf.predict(amcu_arr)
            predicted = predicted.T.reshape(original_shape)
            predicted = np.expand_dims(predicted, axis=0)

            plt.imshow(predicted[0])
            plt.show()

            meta.update({'compress': 'lzw', 'count': 1})
            with rasterio.open(f'{outpath}{region}_rf_predicted.tif', 'w', **meta) as f:
                f.write(predicted)
