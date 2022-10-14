#!/usr/bin/env python3
#cnn_support.py
#REM 2022-10-13

"""
Code for postprocessing of applied CNN models. Use in 'postproc' conda environment.
"""

from skimage import measure

class InstanceSeg():
    """
    Tools for converting from semantic segmentation to instance segmentation
    """

    def __init__(self):
        pass


    def id_instances(self, class_map):
        """
        Convert connected groups of pixels into labelled instances
        """

        instances = measure.label(class_map, connectivity=2)
        return instances


class Ensemble():
    """
    Do ensemble averaging of maps
    WHEN? BEFORE OR AFTER ANY POSTPROCESSING STEPS?
    """
