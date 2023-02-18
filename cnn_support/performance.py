#!/usr/bin/env python3
#performance.py
#REM 2022-02-17

import glob
import numpy as np
import rasterio


class MapManips():
    """
    Methods for manipulating applied model 'maps' - converting probabilities
    to classes, applying NDVI cut, etc.
    """

    def __init__(self, model_output_root):
        self.model_output_root = model_output_root


    def probabilities_to_classes(self, applied_model, threshold_val=0.85, nodata_val=-9999):
        """
        Converts the class probabilities in an applied model tif into
        binary classes using a probability threshold.
        """

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
        print(f'  - Writing {outfile}')
        with rasterio.open(outfile, 'w', **meta) as f:
            f.write(output)


class Evaluate():
    """
    Methods for evaluating model performance and map quality and characteristics
    """
    
    def __init__(self, model_output_root, feature_path, response_path, test_sets):
        self.model_output_root = model_output_root
        self.feature_path = feature_path
        self.response_path = response_path
        self.test_sets = test_sets
        
        
    def _get_correct_and_incorrect_pixels(cls, input_map):
        """
        For a given test region and map thereof, produces an array in which
        correctly-identified building pixels=1, incorrectly-identified pixels=0,
        and all other pixels = NaN.
        """
        
        with rasterio.open(input_map) as f:
            arr = f.read()
            meta = f.meta

        return arr
        
        
    def ndvi_hist(self, model_dir):
        """
        Produces a histogram of NDVI values for threshold maps of test
        regions. Shows values for all pixels (sample thereof), and for pixels
        correctly and incorrectly classified as buildings.
        """

        #for each test_region
        for map_file in glob.glob(f'{model_dir}/*threshold*'):
            
            #get the name of the test region
            test_region = map_file.split('model_')[-1].replace('.tif', '')
            
            #open the map
            with rasterio.open(map_file, 'r') as f:
                map_arr = f.read()
            
            #open the labelled response/reference file, which shows where the buildings are
            ref_file = f'{self.response_path}{self.test_sets[test_region]}_responses.tif'
            with rasterio.open(ref_file, 'r') as f:
                ref_arr = f.read()
                
            #open the NDVI file so we can get NDVI values for the candidates
            ndvi_file = f'{self.feature_path}{self.test_sets[test_region]}_ndvi_hires.tif'
            with rasterio.open(ndvi_file, 'r') as f:
                ndvi_arr = f.read()
                
            print(map_file, ref_file, ndvi_file)
                
            print(map_arr.shape, ref_arr.shape, ndvi_arr.shape)
    