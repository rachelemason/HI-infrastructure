#!/usr/bin/env python3
#figure_code.py
#REM 2022-04-16

"""
Functions for making figures in the HI building mapping paper
"""

import os
import sys
import shutil
import pickle
import numpy as np
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
import matplotlib.image as img
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.lines as lines
import seaborn as sns
from skimage import exposure
import rasterio
import fiona
import pandas as pd
import geopandas as gpd
import sampleraster


BASE_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/'
FIGURE_DATA_PATH = f'{BASE_PATH}/for_figures/'
FIGURE_OUTPUT_PATH = '/home/remason2/figures/'
BAD_BANDS = [list(range(0,5)), list(range(94, 117)), list(range(142, 179)), [211, 212, 213]]


def _make_rgb(img, bands, percentile=2):
    """
    Given the 3D array <img> and a list of three bands, return a color image that can be
    displayed with imshow
    """

    red = img[bands[0], :, :]
    green = img[bands[1], :, :]
    blue = img[bands[2], :, :]

    rgb = np.dstack([red, green, blue])
    rgb = rgb.astype(np.float32) / np.max(rgb)
    
    p_low, p_high = np.percentile(rgb, (percentile, 100 - percentile))
    rgb = exposure.rescale_intensity(rgb, in_range=(p_low, p_high))

    return rgb


def _hillshade(img):
    """
    Return a hillshaded version of img, a numpy array containing a GAO DSM
    """
            
    light = LightSource(azdeg=315, altdeg=45)
    hillshade = light.hillshade(img)
    hillshade = np.expand_dims(hillshade, 0).astype(np.float32)
    
    return hillshade


def model_progress():
    """
    This creates Figue X, which shows the progression of model steps from several 150x150m
    regions
    """
    
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))

    for row, region in enumerate(['MIL2', 'Puako', 'SKonaB', 'KonaMauka', 'CCTrees']):
        for col, data_type in enumerate(['LiDAR', 'model', 'IRGB', 'prob', 'poly']):
            
            ax = axes[row, col]
            
            data_file = f'{FIGURE_DATA_PATH}{region}_{data_type}.tif'
            #read raster data
            if data_type != 'poly':
                with rasterio.open(data_file, 'r') as f:
                    img = f.read()
                    #These images were created by extracting 150m x 150m raster sections in QGIS,
                    #but for some reason the 2m ones don't all have the same shape. Fix that here.
                    if data_type != 'LiDAR' and img.shape[1] > 74:
                        img = img[:, img.shape[1]-74:, :]
                    if data_type != 'LiDAR' and img.shape[2] > 74:
                        img = img[:, :, img.shape[2]-74:]
                
            if data_type == 'LiDAR':
                img = _hillshade(img[0])
                ax.imshow(img[0], cmap='Greys_r')

            if data_type == 'IRGB':
                img = _make_rgb(img, [1, 2, 3])
                ax.imshow(img)

            elif data_type in ['model', 'prob']:
                img = img[0]
                ax.imshow(img, vmin=0, vmax=1, cmap='Greys_r')

            elif data_type == 'poly' and region not in ['KonaMauka', 'CCTrees']:
                
                #first plot the responses (labelled buildings)
                data_file = f'{FIGURE_DATA_PATH}{region}_responses.shp'
                labels = gpd.read_file(data_file)
                style_kwds = {'edgecolor': 'blue', 'linewidth': 2, 'facecolor': 'none'}
                labels.plot(ax=ax, **style_kwds)
                style_kwds = {'edgecolor': 'none','facecolor': 'cyan', 'alpha': 0.5}
                labels.plot(ax=ax, **style_kwds)
                
                #then the model polygons
                data_file = f'{FIGURE_DATA_PATH}{region}_poly.shp'
                gdf = gpd.read_file(data_file)
                style_kwds = {'edgecolor': 'r', 'linewidth': 2, 'facecolor': 'none'}
                gdf.plot(ax=ax, **style_kwds)
                style_kwds = {'edgecolor': 'none', 'facecolor': 'orange', 'alpha': 0.5}
                gdf.plot(ax=ax, **style_kwds)
                
                #add the USBFD polygons
                data_file = f'{FIGURE_DATA_PATH}{region}_bfd.shp'
                bfd = gpd.read_file(data_file)
                style_kwds = {'edgecolor': 'purple', 'linewidth': 2, 'facecolor': 'none', 'ls': ':'}
                bfd.plot(ax=ax, **style_kwds)

            ax.axis('off')
            
    #insert numbers indicating the objects whose spectra were extracted
    axes[0, 2].text(0.11, 0.1, '1', color='magenta', weight='bold', fontsize=16,\
                    transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.76, 0.155, '2', color='magenta', weight='bold', fontsize=16,\
                    transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.5, 0.63, '3', color='magenta', weight='bold', fontsize=16,\
                    transform=axes[0, 2].transAxes)
    axes[1, 2].text(0.46, 0.06, '4', color='magenta', weight='bold', fontsize=16,\
                    transform=axes[1, 2].transAxes)
    axes[4, 2].text(0.28, 0.85, '5', color='magenta', weight='bold', fontsize=16,\
                    transform=axes[4, 2].transAxes)

    #this draws lines separating subplots. Doing it this way because the vector plots
    #have weird boundaries instead of nice squares
    for x in [x*0.1 for x in range(0, 12, 2)]:
        fig.add_artist(lines.Line2D([x, x], [0, 1], color='grey', lw=2))
    for y in [y*0.1 for y in range(0, 12, 2)]:
        fig.add_artist(lines.Line2D([0, 1], [y, y], color='grey', lw=2))

    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.savefig(f'{FIGURE_OUTPUT_PATH}model_progress.png', dpi=450)
    
    
def _identify_water_bands(spectrum, bandval=np.nan):
    """
    Set the spectral regions in <arr> badly affected by telluric H2O to <bandval>. The
    locations of the water bands were first defined by stepping through
    bands in ENVI and looking for the blank images.
    """

    for badbit in BAD_BANDS:
        for band, _ in enumerate(spectrum):
            if band in badbit:
                spectrum[band] = bandval

    return spectrum


def _brightness_norm(arr):
    """
    Brightness normalise a 1D spectrum, i.e., find the vector norm (sum of absolute values
    at each wavelength), and divide all pixels by this number.
    This effectively removes the effect of things like viewing angle that would otherwise
    make some pixels brighter than others regardless of their intrinsic spectra.
    This method also sets regions of poor atmospheric H20 correction to NaN
    (see self.identify_water_bands).
    """

    #We want to exclude telluric H2O regions from calculation, but
    #NaNs and masked arrays don't seem to work with np.linalg.norm, so we
    #temporarily set those spectral regions to 0
    arr = _identify_water_bands(arr, bandval=0)

    norm = arr / scipy.linalg.norm(arr, axis=0)

    #Now set the H20 regions to np.nan
    norm = _identify_water_bands(norm, bandval=np.nan)

    return norm


def extract(feature_file, raster, brightness_norm=True):
    """
    Call sampleraster.py to extract feature spectra from a shapefile, return a
    dictionary of feature spectra.
    Parameters:
        - feature_file, str
          the name of a shapefile containing polygons representing features
          whose spectra are to be extracted
        - raster, str
          the name of the raster from which spectra are to be extracted
    Returns:
        - spectra, dict
          dictionary of feature spectra in which key=feature ID number (arbitrary),
          value=spectrum of that feature.
    """

    #Extract the spectra
    #sampleraster.py outputs a spectrum for every pixel in a feature, and that
    #gets confusing when there are multiple features per shapefile. We could deal
    #with that in postprocessing, but instead we'll handle it here by writing
    #each feature into a separate shapefile and extracting each one separately.
    #Then, the mean spectrum of each feature is calculated and added to a dictionary.

    #read the shapefile that contains the feature(s) to be extracted
    spectra = {}
    features = gpd.read_file(f"{FIGURE_DATA_PATH}/{feature_file}")
    
    raster = f'{FIGURE_DATA_PATH}{raster}'

    #for each feature (polygon) in the shapefile, find the mean spectrum
    #of the corresponding pixels in the reflectance file
    for i in range(len(features)):
        key = (features.loc[i, 'id'], features.loc[i, 'Descr.'])

        #delete existing input and output files/dir - code barfs or hangs if they're present
        if os.path.isfile('temp.shp'):
            os.remove('temp.shp')
        shutil.rmtree('temp', ignore_errors=True)

        #write the polygon for this feature into its own (new) file in the current dir,
        #which sampleraster will use for the extraction of this feature
        features[i: i+1].to_file('temp.shp')

        #extract the spectrum of each pixel in this feature
        sys.argv = ['./sampleraster.py', 'temp.shp', 'temp', raster]
        sampleraster.main()

        #get the spectrum of each pixel, optionally brightness-normalize, add it to a list
        #of spectra, create mean spectrum of all pixels in the feature
        feature_pix = 0 #count number of pixels in feature
        for point in fiona.open("temp/temp.shp"):
            values = []
            #reflectance values are kept in point['properties'] dictionary
            for wav, value in point['properties'].items():
                if wav[0].isdigit(): #find the dict keys that contain band numbers
                    values.append(value) #build up the spectrum of this pixel

            #brightness-normalize the spectrum for this pixel (also takes care of setting
            #water bands to NaN)
            if brightness_norm:
                values = _brightness_norm(values)
            #or just set water bands to NaN
            else:
                _identify_water_bands(values, bandval=np.nan)

            #add the spectrum to any existing spectra for this feature, to build up the
            #summed spectrum
            if key not in spectra:
                spectra[key] = values
            else:
                spectra[key] = [x + y for (x, y) in zip(spectra[key], values)]
            feature_pix += 1

        #divide by the number of pixels in this feature to get the mean spectrum for the feature
        spectra[key] = [y / feature_pix  for y in spectra[key]]

    #Remove the temp directory and anything in it (but usually fails because files are open)
    shutil.rmtree('temp', ignore_errors=True)
    #Remove remaining temp files
    for f in ['temp.shp', 'temp.shx', 'temp.dbf', 'temp.cpg']:
        if os.path.isfile(f):
            os.remove(f)

    return spectra
    
    
def _read_wavelengths(hdr_file):
    """
    Returns the a list containing the wavelength of each band in a reflectance data cube,
    read from its corresponding header file.
    """

    wavelengths = []
    with open(hdr_file, 'r', encoding='utf8') as f:
        read = False
        for line in f:
            if 'wavelength = ' in line:
                read = True
            if read is True and 'wavelength' not in line:
                wavelengths.append(line.split())
                if '}' in line:
                    read = False

    wavelengths = [item.strip(',').strip('}') for sublist in wavelengths for item in sublist]
    wavelengths = [np.round(float(w), 1) for w in wavelengths]

    return wavelengths


def _add_shapley_vals(model_id, wavelengths, ax, label, ptrue=0.01):
    """
    """

    with open(f'{FIGURE_DATA_PATH}shap_{model_id}.pkl', "rb") as f:
        #get the shapley values
        shap = pickle.load(f)
        
    #remove model_prob and tch features (non-spectral)
    shap = shap[:, :-2]
    
    #find mean absolute shap values for each feature (spectral band)
    #also make an array containing indices of top 30 values
    #we'll use these to only plot the most globally-influential features
    shap_maps = np.mean(np.absolute(shap), axis=0)
    #top_features = np.flip(shap_maps.argsort())[:30]

    #sample a sane amount of data (i.e., sample map pixels)
    mask = np.random.choice([False, True], len(shap), p=[1-ptrue, ptrue])
    shap = shap[mask]

    #get the X_train data, edit as for Shapley values
    path = f'{BASE_PATH}bfgn_output_buildings2/model_runs/ensembles/train_test_{model_id}.pkl'
    with open(path, "rb") as f:
         xy = pickle.load(f)
    X_train = xy['X_train']
    X_train = X_train[:, :-2]
    X_train = X_train[mask]

    #re-insert 'bad bands' that were removed for modelling
    #IT IS VERY IMPORTANT THAT BAD_BANDS HERE MATCHES ONES REMOVED FOR MODEL FITTING
    for b in range(len(BAD_BANDS[0])):
        shap = np.insert(shap, 0, np.nan, axis=1)
        X_train = np.insert(X_train, 0, np.nan, axis=1)
        shap_maps = np.insert(shap_maps, 0, 0)
    for b in range(len(BAD_BANDS[1])):
        shap = np.insert(shap, BAD_BANDS[1][0], np.nan, axis=1)
        X_train = np.insert(X_train, BAD_BANDS[1][0], np.nan, axis=1)
        shap_maps = np.insert(shap_maps, BAD_BANDS[1][0], 0)
    for b in range(len(BAD_BANDS[2])):
        shap = np.insert(shap, BAD_BANDS[2][0], np.nan, axis=1)
        X_train = np.insert(X_train, BAD_BANDS[2][0], np.nan, axis=1)
        shap_maps = np.insert(shap_maps, BAD_BANDS[2][0], 0)
    #we don't need to append the three bands that were removed from the long-wavelength end
    
    top_features = np.flip(shap_maps.argsort())[:30]

    ax.axhline(0, color='0.3')

    print('Making the figure')
    #convert band number to wavelength and plot
    for idx, row in enumerate(shap.T):
        #only plot most influential features
        if idx in top_features:
            wav = wavelengths[idx]
            xpos = [wav] * shap.shape[0]
            ax.axvline(xpos[0], color='0.9')
            ax.scatter(xpos, row, s=0.2, c=X_train[:, idx], cmap='cool', zorder=3)

    ax.text(0.02, 0.98, label, ha='left', va='top', transform=ax.transAxes)
    ax.set_xlim(370, 2480)
    ax.set_ylim(-1.49, 1.49) 
    ax.set_xlabel('Wavelength, nm')
    ax.set_ylabel('SHAP value (influence on model output)')
        
    return top_features

    
def _sort_out_stupid_input(dictos):
    """
    I can't even
    """
    d1 = dictos[0]
    d2 = dictos[1]
    d3 = dictos[2]
    d1.update(d2)
    d1.update(d3)
    d1 = {k: d1[k] for k in sorted(d1)}
    
    return d1
    

def shapley_plot(bnorm_spectra, nonorm_spectra, ptrue=0.01):
    """
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    colors = {1:'Black', 2: 'Grey', 3: 'lightsteelblue', 4: 'teal', 5: 'mediumspringgreen'}
    labels = {1: '1. Shingle roof', 2: '2. White roof', 3: '3. Blueish roof',\
              4: '4. Lava rock', 5: '5. Tree'}
    
    wavelengths = _read_wavelengths(f'{FIGURE_DATA_PATH}tile030_mosaic_refl.hdr')
    
    #plot shapley values for model run on brightness-normalized data, in lower subplots
    #do this FIRST so we can get locations of the important features
    features_1 = _add_shapley_vals('run1', wavelengths, ax3, ptrue=ptrue, label='(c)')
    ax3.text(0.97, 0.95, 'Building class', va='center', ha='right', size=8,\
             transform=ax3.transAxes)
    ax3.text(0.97, 0.05, 'Not-building class', va='center', ha='right', size=8,\
             transform=ax3.transAxes)
    
    #shapley values for model run on not-normalized data
    features_2 = _add_shapley_vals('run3', wavelengths, ax4, ptrue=ptrue, label='(d)')
    cmap = plt.cm.get_cmap('cool')
    cbaxes = inset_axes(ax4, width="50%", height="6%", loc='lower right') 
    cbar = fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap), cax=cbaxes, ticks=[],\
                        label='', orientation='horizontal')
    cbar.ax.set_title('Low reflectance            High reflectance', size=8)

    #now plot the spectra in the upper subplots
    plotme = _sort_out_stupid_input(bnorm_spectra)
    for id_num, spectrum in plotme.items():
        ax1.plot(wavelengths, spectrum, color=colors[id_num[0]], zorder=3)
        for idx, wav in enumerate(wavelengths):
            if idx in features_1:
                ax1.axvline(wav, color='0.9')
    ax1.text(0.02, 0.98, '(a)', ha='left', va='top', transform=ax1.transAxes)
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xlim(370, 2480)
    ax1.set_xlabel('Wavelength, nm')
    ax1.set_ylabel('Brightness-normalized reflectance')

    plotme = _sort_out_stupid_input(nonorm_spectra)
    for id_num, spectrum in plotme.items():
        ax2.plot(wavelengths, spectrum, color=colors[id_num[0]], zorder=3,\
                 label=labels[id_num[0]])
        for idx, wav in enumerate(wavelengths):
            if idx in features_2:
                ax2.axvline(wav, color='0.9')
    ax2.text(0.02, 0.98, '(b)', ha='left', va='top', transform=ax2.transAxes)
    ax2.legend(ncol=2, loc='upper right', fontsize=8)
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.set_xlim(370, 2480)
    ax2.set_xlabel('Wavelength, nm')
    ax2.set_ylabel('Reflectance - UNITS?')

    plt.subplots_adjust(hspace=0, left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.savefig(f'{FIGURE_OUTPUT_PATH}shap.png', dpi=450)

    
def histos():
    """
    Get png histograms produced by Performance.ipynb and put them into a single figure
    """
    
    hist1 = img.imread(f'{FIGURE_DATA_PATH}lores_model_hist.png')
    hist2 = img.imread(f'{FIGURE_DATA_PATH}gb_ensemble_prob_hist.png')

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.imshow(hist1)
    ax2.imshow(hist2)
    ax1.text(0.135, 0.86, '(a)', va='top', ha='left', size=10, transform=ax1.transAxes)
    ax2.text(0.135, 0.86, '(b)', va='top', ha='left', size=10, transform=ax2.transAxes)
    
    for ax in (ax1, ax2):
        ax.axis('off')
        
    plt.subplots_adjust(wspace=0, right=0.9)
    plt.savefig(f'{FIGURE_OUTPUT_PATH}histograms.png', dpi=450)
    
    
def median_building_size():
    """
    Use real estate listing stats to estimate the median building size in Hawai'i County
    Data from here: https://www.realtor.com/research/data/
    """
    
    df = pd.read_csv(f'{FIGURE_DATA_PATH}RDC_Inventory_Core_Metrics_County_History.csv',\
                     usecols=['county_name', 'median_square_feet', 'month_date_yyyymm'],\
                     low_memory=False).dropna()
    df = df[df['county_name'].str.contains('hawaii', case=False)]
    
    mean_sq_ft = df['median_square_feet'].median()
    mean_sq_m = mean_sq_ft * 0.0929
    
    print(f'The median median square footage of houses listed for sale in HI county between 2016-2023 is {mean_sq_ft:.0f} sq ft ({mean_sq_m:.0f} sq m)')
    