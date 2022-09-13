#!/usr/bin/env python3
#cnn_support.py
#REM 2022-09-12

"""
Code to support use of the BFGN package (github.com/pgbrodrick/bfg-nets),
which is a package for applying convolutional neural nets to remote
sensing data. Contains classes to help with:
  - setting up and visualizing training data
  - visualizing the applied models
  - looping over different training datasets and
    parameter combinations, and visualizing results
Intended for use with REM's dedicated Jupyter notebooks
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


    def input_files_from_config(self, config_file, print_files=True):
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


    def rasterize(self, reference_raster, shapefile, tif_out, replace_existing):
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


    def extract_raster_section(self, main_file, template, destination, replace_existing=True):
        """
        Write the subset of raster <main_file> specified by <template>
        into <destination> tif.
        """

        def do_the_work():
            print(f'Extracting subset of {main_file} defined by {template}')
            geo = gdal.Open(template)
            ulx, xres, _, uly, _, yres  = geo.GetGeoTransform()
            lrx = ulx + (geo.RasterXSize * xres)
            lry = uly + (geo.RasterYSize * yres)
            gdal.Translate(destination, gdal.Open(main_file), projWin = [ulx, uly, lrx, lry])

        if not replace_existing:
            if os.path.isfile(destination):
                print(f"{destination} already exists, not recreating")
            else:
                do_the_work()
        else:
            do_the_work()


    def tif_to_array(self, tif, get_first_only=False):
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
        arr = arr.astype(np.float32) #int arrays need to be float for the next line to work
        arr[arr == data.GetRasterBand(1).GetNoDataValue()] = np.nan

        return arr


    def boundary_shp_to_mask(self, boundary_file, background_file):
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


    def hillshade(self, ax, to_plot):
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
        mask = self.boundary_shp_to_mask(boundary_file, feature_file)

        # show where the labelled features are, within the training boundaries
        responses = gdal.Open(response_file, gdal.GA_ReadOnly).ReadAsArray()
        mask[responses != 0] = responses[responses != 0] # put the buildings on the canvas
        ax2.imshow(mask)

        plt.tight_layout()


    def plot_training_data(self, features, responses, images_to_plot=3, feature_band=0,\
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


class AppliedModel():
    """
    Methods for visualizing applied models and calculating performance metrics.
    """

    def __init__(self):
        pass


    def probabilities_to_classes(self, method, applied_model_arr, threshold_val=0.95,\
                                 nodata_value=-9999):
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

        return output


    def show_applied_model(self, applied_model, original_img, zoom, responses=None, hillshade=True,\
                          threshold=0.90, filename=None):
        """
        Plots the applied model created by
        bfgn.data_management.apply_model_to_data.apply_model_to_site, converted to a numpy array by
        self.applied_model_as_array. Also zooms into a subregion and shows (1) the applied model
        probabilities, (2) the applied model converted to classes, and (3) the original image.
        """

        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

        if responses is not None:
            shape = gdal.Open(responses, gdal.GA_ReadOnly)
            shape = shape.ReadAsArray()
            shape = np.ma.masked_where(shape < 1.0, shape)

        #Show the applied model probabilities for class 2 over the whole area to which it
        #has been applied
        cbar = ax1.imshow(applied_model[1], cmap='GnBu_r')
        plt.colorbar(cbar, ax=ax1, shrink=0.6, pad=0.01)
        if responses is not None:
            ax1.imshow(shape, alpha=0.7, cmap='Reds_r')

        #Show the area to be zoomed into
        ax1.add_patch(Rectangle((zoom[2], zoom[0]), zoom[3]-zoom[2], zoom[1]-zoom[0],\
                               fill=False, edgecolor='r'))

        #Convert probability map into binary classes using threshold
        classes = self.probabilities_to_classes('threshold', applied_model, threshold)
        ax2.imshow(classes, cmap='Greys')
        if responses is not None:
            ax2.imshow(shape, alpha=0.7, cmap='Reds_r')

        #Zoom into the specified subregion
        ax3.imshow(classes[zoom[0]:zoom[1], zoom[2]:zoom[3]], cmap='Greys')
        #Overplot responses (buildings), if available
        if responses is not None:
            ax3.imshow(shape[zoom[0]:zoom[1], zoom[2]:zoom[3]], alpha=0.8, cmap='viridis_r')

        #The original image for which everything was predicted (same subset/zoom region;
        #only show the first band if there are >1)
        utils = Utils()
        original = utils.tif_to_array(original_img, get_first_only=True)
        #original[original < 0.5] = np.nan #TODO: what was this for?
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
                spc =  ' ' * (18-len(key))
                print(f'{key}{spc}{vals[0]}      {vals[1]}      {vals[2]}      {vals[3]}')
            except AttributeError:
                print(f'{key}    {np.round(vals, 2)}')


    def performance_metrics(self, applied_model, responses, boundary_file):
        """
        Given a model prediction array and a set of responses, calculate precision,
        recall, and f1-score for each class (currently assumes binary classes)
        """

        # convert class probabilities to actual classes
        classes = self.probabilities_to_classes('ML', applied_model)

        # create an array of the same shape as the applied model 'canvas'
        # in which everything outside the training dataset boundary/boundaries is NaN
        utils = Utils()
        mask = utils.boundary_shp_to_mask(boundary_file, responses)

        # insert the labelled responses into the array, inside the training boundaries
        response_array = utils.tif_to_array(responses)
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


class Loops(Utils, AppliedModel):
    """
    Methods for looping over BFGN parameter combinations and evaluating results.
    """

   #using settatr in __init__ is nice but causes pylint to barf, so
   #pylint: disable=no-member

    def __init__(self, application_data, iteration_data):
        self.data_types = None #optionally contained in iteration_data
        for key in application_data:
            setattr(self, key, application_data[key])
        for key in iteration_data:
            setattr(self, key, iteration_data[key])
        self.stats_array = None #defined in _create_stats_array()
        self.default_out = None #defined in loop_over_configs()

        Utils.__init__(self)
        AppliedModel.__init__(self)


    def _train_or_load_model(self, idx, rebuild, fit_model, outdir):
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


    def _apply_model(self, idx, outdir, threshold=0.90):
        """
        Helper method for self.loop_over_configs. Apply an existing trained model
        to data (test fields), write image files, and return performance metrics.
        """

        #retrieve the fitted model
        experiment, data_container = self._train_or_load_model(idx, rebuild=False,\
                                                              fit_model=False, outdir=outdir)

        #for each test region, apply the model and get stats
        stats = []
        with io.capture_output():
            for i, _f in enumerate(self.app_features):

                #find the type of input data used and update application filename to match
                #BFGN needs application data with same number of bands as model training data
                if self.data_types is not None:
                    if self.data_types[idx] in self.parameter_combos[idx][1][0][0]:
                        _f[0] = _f[0].replace('hires_surface', self.data_types[idx])
                        print('Changing filename', _f[0])

                apply_model_to_data.apply_model_to_site(experiment.model, data_container,\
                                                            _f, outdir+self.app_outnames[i])

                applied_model = self.tif_to_array(outdir+self.app_outnames[i]+'.tif')

                #make pdf showing applied model; record stats - only for files with labelled
                #responses
                if self.app_responses[i] is not None:
                    hillshade = bool('res_surface' in _f[0])
                    self.show_applied_model(applied_model, zoom=self.zooms[i],\
                                                original_img=_f[0], hillshade=hillshade,\
                                                responses=self.app_responses[i],\
                                                threshold=threshold,\
                                                filename=f"{outdir}{self.app_outnames[i]}.pdf")

                    stats.append(self.performance_metrics(applied_model, self.app_responses[i],\
                                                self.app_boundaries[i]))

                #create tif containing applied model as binary classes
                temp = self.probabilities_to_classes('threshold', applied_model)
                with rasterio.open(outdir+self.app_outnames[i]+'.tif', 'r') as f:
                    meta = f.meta.copy()
                temp = np.expand_dims(temp, 0)
                temp = temp.astype(np.float32)
                meta.update({'count': 1})
                name = self.app_outnames[i].replace('applied', 'threshold')
                with rasterio.open(outdir+name+'.tif', 'w', **meta) as f:
                    f.write(temp)

                if self.data_types is not None:
                    _f[0] = _f[0].replace(self.data_types[idx], 'hires_surface')

        #store the performance stats (if any) for later use
        with open(f'{outdir}stats.json', 'w', encoding='utf-8') as f:
            json.dump(stats, f)

        return stats


    def _create_stats_array(self, stats):
        """
        Helper method for self.loop_over_configs. Convert big unwieldy list of
        performance metrics created by self.loop_over_configs to self.stats_array
        which is understood by plotting methods.
        """

        #Array to hold the performance metrics
        self.stats_array = np.zeros((len(self.parameter_combos),\
                                     len(self.app_features)*3)) #3 is for the 3 metrics

        for i, _ in enumerate(stats):
            for idx, data in enumerate(stats[i]):
                #stats for class '1.0' (buildings)
                self.stats_array[i, idx*3] = np.round(data['1.0']['precision'], 2)
                self.stats_array[i, idx*3+1] = np.round(data['1.0']['recall'], 2)
                self.stats_array[i, idx*3+2] = np.round(data['1.0']['f1-score'], 2)


    def loop_over_configs(self, rebuild_data=False, fit_model=True, apply_model=True,\
                          use_existing=True, threshold=0.95):
        """
        Loops over BFGN configurations (can be training data or other parameters in the
        config file), and returns a numpy array of model performance metrics (precision,
        recall, F1-score) for all combinations.
        This function intentionally produces much less diagnostic info than if the model
        were fit outside it, using direct calls to BFGN; looping over parameters assumes
        the user already knows what they're doing.
        """

        #Loop over the parameter combos, fit and/or apply model
        all_stats = []
        for idx, params in enumerate(self.parameter_combos):

            #get the current set of parameters
            combo_dict = {}
            for i, j in enumerate(params):
                name = list(self.permutations.keys())[i]
                combo_dict[name] = j

            print('\n===================================================')
            print(f'Working on parameter combination #{idx}:\n')
            for key, value in combo_dict.items():
                print(f'{key}: {value}')

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
                        with open(f'{outdir}stats.json', 'r', encoding='utf-8') as f:
                            stats = json.load(f)
                        print(f'Loaded stats for {outdir} from file')
                    except FileNotFoundError:
                        print(f'Saved stats not found in {outdir}; applying model')
                        stats = self._apply_model(idx, outdir, threshold)
                        plt.close('all')
                else:
                    stats = self._apply_model(idx, outdir, threshold)
                    plt.close('all')
                all_stats.append(stats)

            #Archive the settings file for this parameter combo
            shutil.move(f'new_settings_{idx}.yaml', f'{outdir}new_settings_{idx}.yaml')

        #Once all stats have been gathered, reformat nicely
        if apply_model:
            self._create_stats_array(all_stats)


    def _create_settings_file(self, combo_dict, num):
        """
        Helper method for self.loop_over_configs. Writes the settings file
        needed for the model to run with new parameter or input file settings.
        Returns: nothing.
        """

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


    def parameter_heatmap(self):
        """
        Plots a heatmap of precision, recall and F1 score for a set of model test regions and
        parameter combinations, using <stats_array> produced by the loop_over_configs function.
        """

        print(f'Parameters tested:{list(self.permutations.keys())}')

        _, ax = plt.subplots(figsize=(20, 20))
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
            ax.text(n*3+1, -1, f'Model applied to {region}', ha='center')

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

        fig, _ = plt.subplots(2, 4, figsize=(16, 10))

        for (j, _), ax in zip(enumerate(self.app_outnames), fig.axes):
            if self.app_types[j] == 'neighbour':
                colours = ['lightblue', 'blue', 'black']
            else:
                colours = ['limegreen', 'green', 'darkgreen']

            precision = []
            recall =[]
            f_1 = []
            for i in range(self.stats_array.shape[0]):
                precision.append(self.stats_array[i, j*3])
                recall.append(self.stats_array[i, j*3+1])
                f_1.append(self.stats_array[i, j*3+2])
            ax.plot(range(len(precision)), precision, color=colours[0], marker='o',\
                        ls='-.', label='Precision')
            ax.plot(range(len(recall)), recall, color=colours[1], marker='^',\
                        ls=':', label='Recall')
            ax.plot(range(len(f_1)), f_1, color=colours[2], marker='s', ls='-',\
                        label='F1 score')
            if j == 0:
                ax.legend(loc='lower right')
                ax.set_ylabel('Precision/Recall/F1-score')

            _ = ax.set_xticks(np.arange(len(self.nicknames)), labels=self.nicknames,\
                             rotation=45, ha='right')
            ax.set_xlabel('Model training dataset(s)')

            ax.set_ylim([0, 1])

            title = self.app_outnames[j]
            title = title.split('_')[-1]
            ax.set_title(f'Model applied to {title}')

        plt.tight_layout()

        plt.savefig(self.out_path+'metrics_by_training_data.png', dpi=400)


if __name__ == "__main__":
    pass
