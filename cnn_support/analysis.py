#!/usr/bin/env python3
#analysis.py
#REM 2023-02-10

"""
Code for postprocessing of applied CNN models. Use in 'postproc2' conda environment.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
import rasterio

pd.set_option('display.precision', 2)

DATA_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/labeled_region_features/'
GEO_PATH = '/data/gdcsdata/HawaiiMapping/ProjectFiles/Rachel/geo_data/'


class Analysis():
    """
    Calculate various numbers needed for the mapping paper
    """

    def __init__(self, map_path, analysis_path):
        self.map_path = map_path
        self.analysis_path = analysis_path


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
            gdf = gpd.read_file(f'{self.map_path}/{mapfile}.shp')
            print(f'There are {len(gdf)} buildings in our {mapfile} map')
            count += len(gdf)
        print(f'We identified {count} buildings in total')


    def _create_mapped_region_shpfiles(self, region):
        """
        Helper method for count_msbfd and XX. Writes a shapefile that
        contains a single multipolygon representing the outline of one
        of our geographies, including the NaN (no-data) areas.
        """

        #Use a raster map of the region as a template
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

            mapped_region.to_file(f'{GEO_PATH}{region}_mapped_region', driver='ESRI Shapefile')

            return mapped_region


    def count_msbfd(self):
        """
        Count the number of buildings in the MS Building Footprint Database that are
        within the non-NAN areas of the three DST geographies
        """

        #read the whole MS Building Footprint file for Hawai'i
        msbfd = gpd.read_file(f'{GEO_PATH}Hawaii.geojson')
        msbfd = msbfd.to_crs(epsg=32605)

        count = 0
        for region in ['SKona', 'NKona_SKohala', 'NHilo_Hamakua']:

            if not os.path.exists(f'{GEO_PATH}{region}_mapped_region'):
                mapped_region = self._create_mapped_region_shpfiles(region)
            else:
                mapped_region = gpd.read_file(f'{GEO_PATH}{region}_mapped_region')

            if not os.path.exists(f'{GEO_PATH}MSBFD_{region}'):
                #clip the MSBFD to the mapped region and count remaining buildings
                #save clipped MSBFD file with other shapefiles for future use
                print(f'Clipping {region}')
                msbfd_clip = gpd.clip(msbfd, mapped_region)
                msbfd_clip.to_file(f'{GEO_PATH}MSBFD_{region}', driver='ESRI Shapefile')
            else:
                msbfd_clip = gpd.read_file(f'{GEO_PATH}MSBFD_{region}')

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
        parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_-_Hawaii_County.shp')
        parcels = parcels.to_crs(epsg=32605)

        #read the Hawai'i county Zoning shapefile
        zoning = gpd.read_file(f'{GEO_PATH}Zoning/Zoning_(Hawaii_County).shp')
        zoning = zoning.to_crs(epsg=32605)

        #read the state coastline shapefile
        coast = gpd.read_file(f'{GEO_PATH}Coastline/Coastline.shp')
        coast = coast[coast['isle'] == 'Hawaii'].reset_index(drop=True)
        coast = coast.to_crs(epsg=32605)

        for region in ['NKona_SKohala', 'SKona', 'NHilo_Hamakua']:

            #get the polygon defining the mapped region
            if not os.path.exists(f'{GEO_PATH}{region}_mapped_region'):
                mapped_region = self._create_mapped_region_shpfiles(region)
            else:
                mapped_region = gpd.read_file(f'{GEO_PATH}{region}_mapped_region')

            #get the parcel polygons, clipped to the mapped region
            if not os.path.exists(f'{GEO_PATH}Parcels/Parcels_{region}'):
                print(f'Clipping {region}')
                parcels = gpd.clip(parcels, mapped_region)
                parcels.to_file(f'{GEO_PATH}Parcels/Parcels_{region}',\
                                driver='ESRI Shapefile')
            else:
                parcels = gpd.read_file(f'{GEO_PATH}Parcels/Parcels_{region}')

            print(f'There are {len(parcels)} parcels in {region}')
            parcel_polys = parcels.geometry.tolist()

            #read our building map for the same region
            buildings = gpd.read_file(f'{self.map_path}/{region}.shp')
            print(f'There are {len(buildings)} buildings in {region}')

            #get building centroids, these points will be matched with parcels
            centroids = buildings.geometry.centroid

            #do the matching
            spatial_index = centroids.sindex
            #need to have at least one column specified to avoid weird problem
            #with multipolygons...
            matches = gpd.GeoDataFrame(columns=['Parcel'])
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

            #also save to a shapefile for this region
            matches.to_file(f'{self.analysis_path}{region}_parcel_occupation.shp',\
                            driver='ESRI Shapefile')

            open_parcels = len(matches[matches['Occupied'] == '0'])
            print(f'There are approximately {open_parcels} undeveloped parcels in {region}')
            print('(Some parcels contain multiple buildings)')


    def parcel_properties(self):
        """
        Make a stacked histogram showing occupied and unoccupied parcels for
        each of the 3 geographies
        """

        _ = plt.figure()
        for n, region in enumerate(['SKona', 'NKona_SKohala', 'NHilo_Hamakua']):
            data = gpd.read_file(f'{self.analysis_path}{region}_parcel_occupation.shp')

            print(f'There are {len(data)} occupied parcels in our map for {region}')
            print(data['County Bld'].unique())
            county_unknown = len(data[(data['Occupied'] == '1') & (data['County Bld'] == '0')])
            print(f'- {county_unknown} occupied parcels have no county building value')
            county_known = len(data[(data['Occupied'] == '0') & (data['County Bld'] == '1')])
            print(f'- {county_known} parcels have building values but are unoccupied according to our maps')

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


    def parcel_plot(self, column, binwidth=200, xmax=5, ymax=1000):
        """
        Make a multi-panel set of histograms showing n as a function of <column>,
        where <column> is a column in the parcel occupation file. For example,
        show number of parcels as a function of distance from the coast. One panel
        per geography.
        """

        fig, _ = plt.subplots(2, 2, figsize=(16, 8))

        for region, ax in zip(['SKona', 'NKona_SKohala', 'NHilo_Hamakua'], fig.axes):
            data = gpd.read_file(f'{self.analysis_path}{region}_parcel_occupation.shp')
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
