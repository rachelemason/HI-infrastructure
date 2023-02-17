#!/usr/bin/env python3
#run_models.py
#REM 2022-02-17

"""
Code to apply CNNs to GAO LiDAR data using the BFGN package (github.com/pgbrodrick/bfg-nets),
Intended for use with the RunModel.ipynb Jupyter Notebook. Requires bfgn-gpu
environment.
"""

import os
import shutil
import warnings
import glob
from IPython.utils import io
import numpy as np
import matplotlib.pyplot as plt
import fiona
import rasterio

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


class TrainingData():
    """
    Methods for visualizing training data.
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
    def create_training_lists(cls, paths, all_training_sets, desired_training_set):
        """
        Return a properly-formatted list of lists of training data, based on the available
        training data and the desired subset.
        """

        parameter_combos = []
        for train in desired_training_set:
            bound = [f"{paths['boundary']}{all_training_sets[item]}_boundary.shp"\
                          for item in train]
            feat = []
            resp = []
            for item in train:
                feat.append([f"{paths['features']}{all_training_sets[item]}_hires_surface.tif"])
                resp.append([f"{paths['responses']}{all_training_sets[item]}_responses.tif"])
            parameter_combos.append([bound, feat, resp])

        return parameter_combos


class Loops():
    """
    Methods for looping over BFGN parameter combinations
    """

   #using settatr in __init__ is nice but causes pylint to barf, so
   #pylint: disable=no-member

    def __init__(self, iteration_data, settings_file):
        self.data_types = [] #optionally filled in in iteration_data
        for key in iteration_data:
            setattr(self, key, iteration_data[key])
        self.settings_file = settings_file


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


    def loop_over_configs(self, rebuild_data=False, fit_model=True, use_existing=False):
        """
        Loops over BFGN configurations (can be training data or other parameters in the
        config file)
        """

        #Loop over the parameter combos, fit model
        for idx, params in enumerate(self.parameter_combos):
            #get the current set of parameters
            combo_dict = {}
            for i, j in enumerate(params):
                name = list(self.permutations.keys())[i]
                combo_dict[name] = j

            print('\n===================================================')
            print(f'Working on parameter combination #{idx}:\n')

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

            #Archive the settings file for this parameter combo
            shutil.move(f'new_settings_{idx}.yaml', f'{outdir}new_settings_{idx}.yaml')


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


    def apply_model(self, config_file, application_file, outfile):
        """
        Apply an existing model to a new area and write to file.
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
