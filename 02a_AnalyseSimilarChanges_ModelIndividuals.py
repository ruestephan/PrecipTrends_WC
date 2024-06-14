# -*- coding: utf-8 -*-
'''
date 07/06/2024
@author: Ruth Stephan
'''


#packages:
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import seaborn as sns
import cartopy.feature as cfeature
import copy
import pickle



#paths:
os.getcwd()
data_path = '/mnt/share/scratch/rs1155/data/CMIP6/output_data/'
plot_path = '/mnt/share/scratch/rs1155/plots/model_individuals/'

# open model_mean for precipitation change:
mmean_output_path = '/mnt/share/scratch/rs1155/data/CMIP6/model_means/'
modelmean = xr.open_dataset(mmean_output_path + 'mean_combined.nc')

def calc_vardiff(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    
    #data = sdelta_ds
    #var = 'pr'
    
    var20002025 = data[var].sel(time=slice('2000', '2025')).mean(dim='time')
    var20752100 = data[var].sel(time=slice('2075', '2100')).mean(dim='time')
       
    vardiff21002025 = var20752100-var20002025
    
    #vardiff21002025.abs() >= var20002025*0.1
    
    return vardiff21002025

def calc_vardiff_season(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    svar20002025 = data[var].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
    svar20752100 = data[var].sel(time=slice('2075', '2100')).groupby('time.season').mean(dim='time')
   
    svardiff21002025 = svar20752100-svar20002025
    
    
    return svardiff21002025

yearly_modelmean_dpr = calc_vardiff(data = modelmean, var='pr')
season_modelmean_dpr = calc_vardiff_season(data = modelmean, var='pr')

# Set several styles for plotting:
sns.set_style('ticks')
plt.rcParams['font.family'] = 'DejaVu Sans'
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '110m', edgecolor='face', facecolor='white')
pal_PRGn = mpl.colormaps.get_cmap('PRGn') 
pal_PRGn.set_bad(color='lightgray')
pal_coolwarm = mpl.colormaps.get_cmap('coolwarm') 
pal_coolwarm.set_bad(color='lightgray')
pal_coolwarm_r = mpl.colormaps.get_cmap('coolwarm_r') 
pal_coolwarm_r.set_bad(color='lightgray')
pal_RdBu = mpl.colormaps.get_cmap('RdBu') 
pal_RdBu.set_bad(color='lightgray')
pal_Reds = mpl.colormaps.get_cmap('Reds') 
pal_Reds.set_bad(color='lightgray')
pal_Greens = mpl.colormaps.get_cmap('Greens') 
pal_Greens.set_bad(color='lightgray')
pal_Purples = mpl.colormaps.get_cmap('Purples') 
pal_Purples.set_bad(color='lightgray')
pal_Greens_r = mpl.colormaps.get_cmap('Greens_r') 
pal_Greens_r.set_bad(color='lightgray')
pal_Reds_r = mpl.colormaps.get_cmap('Reds_r')  
pal_Reds_r.set_bad(color='lightgray')
colors_highest_sim = ['forestgreen', 'skyblue', 'darkgoldenrod']
cmap_highest_sim = ListedColormap(colors_highest_sim)
cmap_highest_sim.set_bad('lightgray')
colors_highest_sim_Pmask = ['forestgreen', 'steelblue', 'darkgoldenrod', 'lightgreen', 'skyblue','wheat'] 
cmap_highest_sim_Pmask = ListedColormap(colors_highest_sim_Pmask)
cmap_highest_sim_Pmask.set_bad('lightgray')



'''
Functions:
'''
#import functions from the subfolder scripts
#from Scripts.Functions.defined_functions import  calc_vardiff, calc_vardiff_season, calc_vardiff_yearmonth


# Calculation of an xarray that stores the area info (km2) in each cell:
def calculate_cell_area(lat, lon):
    """
    Calculate the area of each cell in a latitude-longitude grid.

    Parameters:
    - lat: Array of latitudes.
    - lon: Array of longitudes.

    Returns:
    - area: Array of areas in square kilometers.
    """
    R = 6371.0  # Radius of the Earth in kilometers
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    #lat_diff = np.abs(np.diff(lat_rad))
    lon_diff = np.abs(np.diff(lon_rad))
    
    # Calculate the area for each cell
    areas = np.zeros((len(lat), len(lon)))
    
    for i in range(len(lat) - 1):
        for j in range(len(lon) - 1):
            areas[i, j] = (R ** 2) * np.abs(np.sin(lat_rad[i]) - np.sin(lat_rad[i + 1])) * lon_diff[j]
    
    # Handle the last row and column
    areas[-1, :] = areas[-2, :]
    areas[:, -1] = areas[:, -2]
    
    return areas


def find_lower_index(arr1, arr2, arr3):
    """
    Get the lowest value of three arrays and store that as in index in a new array.
    """

    # Mask NaN values in all arrays
    mask_arr1 = np.isnan(arr1)
    mask_arr2 = np.isnan(arr2)
    mask_arr3 = np.isnan(arr3)
    
    # Create masked arrays to handle NaN values
    arr1_masked = np.ma.masked_array(arr1, mask_arr1)
    arr2_masked = np.ma.masked_array(arr2, mask_arr2)
    arr3_masked = np.ma.masked_array(arr3, mask_arr3)
    
    # Find the minimum value among non-NaN values
    min_arr = np.minimum(abs(arr1_masked), np.minimum(abs(arr2_masked), abs(arr3_masked)))
    
    # Determine which array has the minimum value
    index_arr = np.where(min_arr == abs(arr1_masked), 1, np.where(min_arr == abs(arr2_masked), 2, 3)).astype(float)
    
    # Set NaN values back to NaN
    index_arr[mask_arr1] = np.nan
    index_arr[mask_arr2] = np.nan
    index_arr[mask_arr3] = np.nan
    
    return index_arr


def high_sim_allcells(data, var1, var2, var3):
    """
    Applies find_lower_index on all cells
    """

    input_vars = [abs(data[var1]), abs(data[var2]), abs(data[var3])]
    
    high_results = xr.apply_ufunc(
        find_lower_index,
        *input_vars,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        dask='parallelized',
        output_dtypes=[float]
    )
    
    return high_results


def ratio1_index(arr1, arr2, arr3, threshold=0.1):
    """
    Get the value closest to 1 of three arrays and store that as in index in a new array. 
    Threshhold is used to identify if difference between closest ratio is less then xy % to 2nd closest value.
    If this is the case the array gets the index 4.
    """

    # Mask NaN values in all arrays
    mask_arr1 = np.isnan(arr1)
    mask_arr2 = np.isnan(arr2)
    mask_arr3 = np.isnan(arr3)
    
    # Create masked arrays to handle NaN values
    arr1_masked = np.ma.masked_array(arr1, mask_arr1)
    arr2_masked = np.ma.masked_array(arr2, mask_arr2)
    arr3_masked = np.ma.masked_array(arr3, mask_arr3)
    
    # Calculate the absolute differences from 1, ignoring NaNs
    diff1 = abs(arr1_masked - 1)
    diff2 = abs(arr2_masked - 1)
    diff3 = abs(arr3_masked - 1)
    
    # Find the minimum differences
    min_diff = np.minimum(diff1, np.minimum(diff2, diff3))
    
    # Find the value(s) closest to 1
    closest_to_1 = np.where(diff1 == min_diff, arr1_masked, 
                            np.where(diff2 == min_diff, arr2_masked, arr3_masked))
    
    # Determine which array has the minimum value
    index_arr = np.where(closest_to_1 == arr1_masked, 1, 
                         np.where(closest_to_1 == arr2_masked, 2, 3)).astype(float)
    
    
    ''' Do that with the model agreement
    #Analyse whether the highest similarity too close to the 2nd highest similarity:
    # 1. get the second closest to 1:
    second_diff = np.where(diff1 == min_diff, np.minimum(diff2, diff3),
                           np.where(diff2 == min_diff, np.minimum(diff1, diff3), np.minimum(diff1, diff2)))
    # 2. get the difference between the 1st and 2nd closest and the calculate the ratio of the difference and the 1st closest
    ratio = np.abs(min_diff - abs(second_diff)) / np.abs(min_diff)
    
    # 3. When this ratio is lower than 50 %, meaning that the 2nd closest to 1 needs to at least deviate with 50 % from the first --> unclear result:
    unclear = ratio < threshold
    index_arr[unclear] = 4
    '''
    
    # Set NaN values back to NaN
    index_arr[mask_arr1] = np.nan
    index_arr[mask_arr2] = np.nan
    index_arr[mask_arr3] = np.nan
    
    return index_arr


def ratio1_index_allcells(data, var1, var2, var3):
    """
    Applies ratio1_index on all cells
    """

    input_vars = [data[var1], data[var2], data[var3]]
    
    ratio_results = xr.apply_ufunc(
        ratio1_index,
        *input_vars,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        dask='parallelized',
        output_dtypes=[float]
    )
    
    return ratio_results


def calc_vardiff(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    
    #data = sdelta_ds
    #var = 'pr'
    
    var20002025 = data[var].sel(time=slice('2000', '2025')).mean(dim='time')
    var20752100 = data[var].sel(time=slice('2075', '2100')).mean(dim='time')
       
    vardiff21002025 = var20752100-var20002025
    
    #vardiff21002025.abs() >= var20002025*0.1
    
    return vardiff21002025

def calc_vardiff_season(data, var):
    """
    calculates the difference between period 2075-2100 and 2000-2025
    """
    svar20002025 = data[var].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
    svar20752100 = data[var].sel(time=slice('2075', '2100')).groupby('time.season').mean(dim='time')
   
    svardiff21002025 = svar20752100-svar20002025
    
    
    return svardiff21002025



'''
Loop over all models to identify most similar changes with difference and ratio:
'''


# Get to the files that should be uses:
files_in_directory = os.listdir(data_path)
combined_files = [file for file in files_in_directory if 'combined' in file]
model_names = [filename.split('_combined.nc')[0] for filename in combined_files]    

# Get masks: 
path_masks = '/mnt/share/scratch/rs1155/data/CMIP6/'
yearly_precip_threshold = xr.open_dataset(path_masks + 'YearlyPrecip_Mask.nc')
yearly_change_greater = xr.open_dataset(path_masks + 'YearlyPrecipChange_Mask.nc')
seasons_change_greater = xr.open_dataset(path_masks + 'SeasonsPrecip_Mask.nc')


all_models_delta = []
all_models_sdelta = []

for model_name in model_names:
    
    #model_name = model_names[4]
    
    combined = xr.open_dataset(data_path + model_name + '_combined.nc')
    combined = combined.sel(time=slice('2000', '2100'))
    

    # Mask out areas with too little total precipitation and to little change in precipitation:
    #combined['pr'] = combined['pr'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    #combined['evspsbl'] = combined['evspsbl'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    #combined['mrro'] = combined['mrro'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    #combined['mrso'] = combined['mrso'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    #combined['PET'] = combined['PET'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    #combined['PETR'] = combined['PETR'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    
    
    # calculate area for each gricell and store that in a 2nd array:
    latitudes = combined['lat'].values
    longitudes = combined['lon'].values
    cell_areas = calculate_cell_area(latitudes, longitudes)
    cell_areas_da = xr.DataArray(cell_areas, dims=['lat', 'lon'], coords={'lat': latitudes, 'lon': longitudes})
    
 
    ###################################### 
    ###         annual trend           ###
    ######################################  
    
    delta_ds = combined.copy()   
    
    delta_ds = delta_ds.assign(d_pr = calc_vardiff(delta_ds, 'pr'))
    delta_ds = delta_ds.assign(d_evspsbl = calc_vardiff(delta_ds, 'evspsbl'))
    delta_ds = delta_ds.assign(d_mrro = calc_vardiff(delta_ds, 'mrro'))
    delta_ds = delta_ds.assign(d_PETR = calc_vardiff(delta_ds, 'PETR'))
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = delta_ds.copy()
    plot_dt['d_pr'] = plot_dt['d_pr'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['d_evspsbl'] = plot_dt['d_evspsbl'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['d_mrro'] = plot_dt['d_mrro'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['d_PETR'] = plot_dt['d_PETR'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    # Plot the data
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    variables = ['d_pr', 'd_evspsbl', 'd_mrro', 'd_PETR']
    titles = [r'$\Delta P$', r'$\Delta ET$', r'$\Delta Q$', r'$\Delta RES$']
    for (v, title, ax) in zip(variables, titles, axes.flat):
        if v in ['d_PETR']:
            plot_dt[v].plot.imshow(ax=ax, cmap=pal_PRGn, extend='both',
                                    center=0,
                                    vmin=-5e-7,
                                    vmax=5e-7,
                                    cbar_kwargs={'label': '[kg m⁻² s⁻¹]'})
        else:
            plot_dt[v].plot.imshow(ax=ax, cmap=pal_PRGn, extend='both',
                                    center=0,
                                    vmin=-5e-6,
                                    vmax=5e-6,
                                    cbar_kwargs={'label': '[kg m⁻² s⁻¹]'})
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_Delta_Vars_Yearly.png', dpi=300, transparent=True)
    plt.show()
    
    #################################
    # High Similarity - Differences #
    #################################
    
    # substract the changes from each other: 
    delta_ds['delta_PET'] = delta_ds['d_pr']-delta_ds['d_evspsbl']     
    delta_ds['delta_PR'] = delta_ds['d_pr']-delta_ds['d_mrro']     
    delta_ds['delta_PPETR'] = delta_ds['d_pr']-delta_ds['d_PETR']   
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = delta_ds.copy()
    plot_dt['delta_PET'] = plot_dt['delta_PET'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['delta_PR'] = plot_dt['delta_PR'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['delta_PPETR'] = plot_dt['delta_PPETR'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    
    variables = ['delta_PET', 'delta_PR', 'delta_PPETR']
    titles_difference = [r'$\Delta (P-ET)$', r'$\Delta (P-Q)$', r'$\Delta (P-RES)$']
    for (v, title, ax) in zip(variables, titles_difference, axes.flat): 
        abs(plot_dt[v]).plot.imshow(ax=ax, cmap=pal_Reds_r, extend='both',
                                #center=0,
                                #vmin=-4e-6,
                                vmin=0,
                                vmax=4e-6,
                                cbar_kwargs={'label': '[kg m⁻² s⁻¹]'})
        
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_Difference_PVars_Yearly.png', dpi=300, transparent=True)
    plt.show()
    
    
    
    ################ Check most similar change:
    sim_change = high_sim_allcells(data=delta_ds, var1='delta_PET', var2='delta_PR', var3='delta_PPETR')
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = sim_change.copy()
    plot_dt = plot_dt.where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
 
    
    # Calculate areas for each category
    PET_area = float((cell_areas_da * (plot_dt == 1)).sum())
    PR_area = float((cell_areas_da * (plot_dt == 2)).sum())
    PPETR_area = float((cell_areas_da * (plot_dt == 3)).sum())
    
    # Total area
    total_area = PET_area + PR_area + PPETR_area
    
    # Calculate percentage of area covered
    perc_PET = (PET_area / total_area) * 100
    perc_PR = (PR_area / total_area) * 100
    perc_PPETR = (PPETR_area / total_area) * 100
    
    # Bar plot data
    categories = ['delta_PET', 'delta_PR', 'delta_PPETR']
    values = [perc_PET, perc_PR, perc_PPETR]
    
    ## Plot
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
    
    # Plot the main map
    plot_dt.plot.imshow(ax=axes, cmap=cmap_highest_sim, extend='both', vmin=1, vmax=3)
    axes.set_title('Highest Similarity - Yearly')
    
    # Add the ocean feature colored in white
    axes.add_feature(ocean)
    axes.coastlines()
    
    left, bottom, width, height = 0.13, 0.16, 0.10, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    ax_inset.bar(categories, values, color=colors_highest_sim, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('Area [%]')
    ax_inset.set_yticks([0, 20, 40, 60])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Difference_Yearly.png', dpi=300, transparent=True)
    plt.show()
    
    
    ### analyse in addition if the Precipitation will increase or decrease:
    m_neg = yearly_modelmean_dpr < 0
    
    np.unique(sim_change)
    data_modified = sim_change.where(~m_neg, sim_change+3)
    np.unique(data_modified)
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = data_modified.copy()
    plot_dt = plot_dt.where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    #Barplot global:
    PET_area_pos = float((cell_areas_da * (plot_dt == 1)).sum())
    PR_area_pos = float((cell_areas_da * (plot_dt == 2)).sum())
    PPETR_area_pos = float((cell_areas_da * (plot_dt == 3)).sum())
    
    PET_area_neg = float((cell_areas_da * (plot_dt == 4)).sum())
    PR_area_neg = float((cell_areas_da * (plot_dt == 5)).sum())
    PPETR_area_neg = float((cell_areas_da * (plot_dt == 6)).sum())
    
    # Total area
    total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg
    
    perc_PET_pos = (PET_area_pos / total_area) * 100
    perc_PR_pos = (PR_area_pos / total_area) * 100
    perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    
    perc_PET_neg = (PET_area_neg / total_area) * 100
    perc_PR_neg = (PR_area_neg / total_area) * 100
    perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    
    categories = ['$\Delta$ET-$\Delta$P pos', '$\Delta$R-$\Delta$P pos', '$\Delta$(P-ET-R)-$\Delta$P pos', 
                  '$\Delta$ET-$\Delta$P neg', '$\Delta$R-$\Delta$P neg', '$\$\Delta$(P-ET-R)-$\Delta$P neg']
    values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, 
              perc_PET_neg, perc_PR_neg, perc_PPETR_neg]
    
    
    ## Plot:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
    
    # Plot the main map
    plot_dt.plot.imshow(ax=axes, cmap=cmap_highest_sim_Pmask, extend='both', vmin=1, vmax=6)
    axes.add_feature(ocean)
    axes.coastlines()
    axes.set_title('Highest Similarity - Yearly')
    
    # Inset for the bar plot
    left, bottom, width, height = 0.11, 0.16, 0.14, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    ax_inset.bar(categories, values, color=colors_highest_sim_Pmask, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('Area [%]')
    ax_inset.set_yticks([0, 10, 20, 30, 40])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Difference_Yearly_Pmask.png', dpi=300, transparent=True)
    plt.show()
    
    
    ############################
    # High Similarity - RATIOS #
    ############################ 
    
    delta_ds['ratio_ETP'] = delta_ds['d_evspsbl']/delta_ds['d_pr']
    delta_ds['ratio_RP'] = delta_ds['d_mrro']/delta_ds['d_pr'] 
    delta_ds['ratio_PETRP'] = delta_ds['d_PETR']/delta_ds['d_pr']
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = delta_ds.copy()
    plot_dt['ratio_ETP'] = plot_dt['ratio_ETP'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['ratio_RP'] = plot_dt['ratio_RP'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])
    plot_dt['ratio_PETRP'] = plot_dt['ratio_PETRP'].where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    
    variables = ['ratio_ETP', 'ratio_RP', 'ratio_PETRP']
    titles_ratio = titles = [r'$\Delta ET / \Delta P $', r'$\Delta Q / \Delta P $', r'$\Delta RES / \Delta P $']
    
    for (v, title, ax) in zip(variables, titles_ratio, axes.flat):  # Verwende zip() anstelle von enumerate()
        plot_dt[v].plot.imshow(ax=ax, 
                                cmap=pal_RdBu, 
                                extend='both',
                                #center=0,
                                #vmin=-4e-6,
                                vmin=-2,
                                vmax=2)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Ratio_Yearly.png', dpi=300, transparent=True)
    plt.show()
    
    
    ratio1 = ratio1_index_allcells(data=delta_ds, var1='ratio_ETP', var2='ratio_RP', var3='ratio_PETRP')
    print(np.unique(ratio1))
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = ratio1.copy()
    plot_dt = plot_dt.where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    
    ############## Plot most similar changes:
    PET_area = float((cell_areas_da * (plot_dt == 1)).sum())
    PR_area = float((cell_areas_da * (plot_dt == 2)).sum())
    PPETR_area = float((cell_areas_da * (plot_dt == 3)).sum())
    
    total_area = PET_area + PR_area + PPETR_area #+ unclear_area
    
    perc_PET = (PET_area / total_area) * 100
    perc_PR = (PR_area / total_area) * 100
    perc_PPETR = (PPETR_area / total_area) * 100
    
    categories = ['ratio_ETP', 'ratio_RP', 'ratio_PETRP']
    values = [perc_PET, perc_PR, perc_PPETR]
    
    
    ## Plot:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
    
    # Plot the main map
    plot_dt.plot.imshow(ax=axes, cmap=cmap_highest_sim, extend='both', vmin=1, vmax=3)
    axes.add_feature(ocean)
    axes.coastlines()
    axes.set_title('Highest Similarity - Yearly')
    
    # Inset for the bar plot
    left, bottom, width, height = 0.13, 0.16, 0.10, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    ax_inset.bar(categories, values, color=colors_highest_sim, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('Area [%]')
    ax_inset.set_yticks([0, 20, 40, 60])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Ratio_Yearly.png', dpi=300)
    plt.show()
    
    ### analyse in addition if the Precipitation will increase or decrease:
    m_neg = yearly_modelmean_dpr < 0
    
    print(np.unique(ratio1))
    data_modified = ratio1.where(~m_neg, ratio1 + 3)
    print(np.unique(data_modified))
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = data_modified.copy()
    plot_dt = plot_dt.where(yearly_precip_threshold['pr'] & yearly_change_greater['pr'])

    
    #Barplot global:
    PET_area_pos = float((cell_areas_da * (plot_dt == 1)).sum())
    PR_area_pos = float((cell_areas_da * (plot_dt == 2)).sum())
    PPETR_area_pos = float((cell_areas_da * (plot_dt == 3)).sum())
    
    PET_area_neg = float((cell_areas_da * (plot_dt == 4)).sum())
    PR_area_neg = float((cell_areas_da * (plot_dt == 5)).sum())
    PPETR_area_neg = float((cell_areas_da * (plot_dt == 6)).sum())
    
    # Total area
    total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg 
    
    perc_PET_pos = (PET_area_pos / total_area) * 100
    perc_PR_pos = (PR_area_pos / total_area) * 100
    perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    
    perc_PET_neg = (PET_area_neg / total_area) * 100
    perc_PR_neg = (PR_area_neg / total_area) * 100
    perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    
    
    categories = ['$\Delta$ET/$\Delta$P pos', '$\Delta$R/$\Delta$P pos', '$\Delta$RES/$\Delta$P pos',
                  '$\Delta$ET/$\Delta$P neg', '$\Delta$R/$\Delta$P neg', '$\$\Delta$RES/$\Delta$P neg']
    values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos,
              perc_PET_neg, perc_PR_neg, perc_PPETR_neg]
    
    
    
    ## Plot:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        
    
    # Plot the main map
    plot_dt.plot.imshow(ax=axes, cmap=cmap_highest_sim_Pmask,  extend='both', 
                                          cbar_kwargs={'label': 'Highest Change'}, 
                                          vmin=1, vmax=6)
    axes.add_feature(ocean)
    axes.coastlines()
    axes.set_title('Highest Similarity - Yearly')
    
    # Inset for the bar plot
    left, bottom, width, height = 0.11, 0.16, 0.14, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    ax_inset.bar(categories, values, color=colors_highest_sim_Pmask, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('Area [%]')
    ax_inset.set_yticks([0, 10, 20, 30, 40])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Ratio_Yearly_Pmask.png', dpi=300)
    plt.show()
    
    
    
    # collect and save selected data:
    delta_ds['sim_change'] = ratio1
    delta_ds['sim_change_Pmask'] = data_modified
    delta = delta_ds[['d_pr', 'd_mrro', 'd_evspsbl', 'd_PETR',
                      'ratio_ETP', 'ratio_RP', 'ratio_PETRP', 'sim_change', 'sim_change_Pmask']]
    delta.to_netcdf(data_path + model_name + '_delta.nc')
    
    #all_models_delta.append(delta)
    all_models_delta.append({'name': model_name, 'delta': delta})
    
    print(f'stored delta data for model {model_name}.')
    ###################################### 
    ###        seasonal changes        ###
    ###################################### 
    
    sdelta_ds = combined.copy() 
    
    # Calculate Difference between 2100 and 2025
    sdelta_ds = sdelta_ds.assign(d_tas = calc_vardiff_season(sdelta_ds, 'tas'))
    sdelta_ds = sdelta_ds.assign(d_pr = calc_vardiff_season(sdelta_ds, 'pr'))
    sdelta_ds = sdelta_ds.assign(d_evspsbl = calc_vardiff_season(sdelta_ds, 'evspsbl'))
    sdelta_ds = sdelta_ds.assign(d_mrro = calc_vardiff_season(sdelta_ds, 'mrro'))
    sdelta_ds = sdelta_ds.assign(d_PETR = calc_vardiff_season(sdelta_ds, 'PETR'))
    sdelta_ds = sdelta_ds.assign(d_mrso = calc_vardiff_season(sdelta_ds, 'mrso'))
    
    #sdelta_ds['d_pr'] = sdelta_ds['d_pr'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    #sdelta_ds['d_evspsbl'] = sdelta_ds['d_evspsbl'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    #sdelta_ds['d_mrro'] = sdelta_ds['d_mrro'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    #sdelta_ds['d_PETR'] = sdelta_ds['d_PETR'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    #sdelta_ds['d_mrso'] = sdelta_ds['d_mrso'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    #sdelta_ds['d_tas'] = sdelta_ds['d_tas'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])

    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = sdelta_ds.copy()
    plot_dt['d_pr'] = plot_dt['d_pr'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['d_evspsbl'] = plot_dt['d_evspsbl'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['d_mrro'] = plot_dt['d_mrro'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['d_PETR'] = plot_dt['d_PETR'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])

    variables = ['d_pr','d_evspsbl','d_mrro', 'd_PETR']
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    
    fig, axes = plt.subplots(nrows=len(seasons), ncols=len(variables), figsize=(20, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    titles = [r'$\Delta P$', r'$\Delta ET$', r'$\Delta Q$', r'$\Delta RES$']
    
    for i, s in enumerate(seasons):
        for j, title, variable in zip([0,1,2,3],titles,variables):
            #season ='DJF'
            #variable = 'd_pr'
            ax = axes[i, j]
            plot_dt[variable].sel(season=s).plot.imshow(ax=ax, cmap=pal_PRGn, extend='both',
                                                    center=0,
                                                    vmin=-5e-6,
                                                    vmax=+5e-6,
                                                    cbar_kwargs={'label': '[kg m⁻² s⁻¹]'})
            ax.add_feature(ocean)
            ax.coastlines()
            ax.set_title(f'{title} - {s}')
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_Delta_Vars_Seasons.png', dpi=300, transparent=True)
    plt.show()
    
    
    #################################
    # High Similarity - Differences #
    #################################
    
    # substract the changes from each other: 
    sdelta_ds['delta_PET'] = sdelta_ds['d_pr']-sdelta_ds['d_evspsbl']     
    sdelta_ds['delta_PR'] = sdelta_ds['d_pr']-sdelta_ds['d_mrro']     
    sdelta_ds['delta_PPETR'] = sdelta_ds['d_pr']-sdelta_ds['d_PETR']    
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = sdelta_ds.copy()
    plot_dt['delta_PET'] = plot_dt['delta_PET'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['delta_PR'] = plot_dt['delta_PR'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['delta_PPETR'] = plot_dt['delta_PPETR'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])

    
    fig, axes = plt.subplots(nrows=len(seasons), ncols=3, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    variables = ['delta_PET', 'delta_PR', 'delta_PPETR']
    for i, s in enumerate(seasons):
        for j, title, variable in zip([0,1,2,3], titles_difference, variables):
            ax = axes[i, j]
            abs(plot_dt[variable]).sel(season=s).plot.imshow(ax=ax, cmap=pal_Reds_r, extend='both',
                                                    #center=0,
                                                    #vmin=-5e-6,
                                                    vmin=0,
                                                    vmax=7e-6,
                                                    cbar_kwargs={'label': f'{variable}'})
            ax.coastlines()
            ax.set_title(f'{title} - {s}')
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_Difference_PVars_Seasons.png', dpi=300, transparent=True)
    plt.show()
    
    high_sim_seasons = {}
    for s in seasons: 
        #highest_sim = high_sim_allcells(data = sdelta_ds.where(sdelta_ds['season'] ==s, drop=True), var1='delta_PET', var2 = 'delta_PR')
        highest_sim = high_sim_allcells(data = sdelta_ds.where(sdelta_ds['season'] ==s, drop=True), var1='delta_PET', var2 = 'delta_PR', var3= 'delta_PPETR')
        high_sim_seasons[s] = highest_sim
        
    #get rid of season dimension:
    high_sim_seasons = {s: da.squeeze('season') for s, da in high_sim_seasons.items()}
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = high_sim_seasons.copy()
    plot_dt = {season: plot_dt[season].where(yearly_precip_threshold['pr']) for season in plot_dt}
    plot_dt = {
        season: plot_dt[season].where(seasons_change_greater['pr'].sel(season=season))
        for season in plot_dt}

    
    bar_data = []
    for i in seasons:
        PET_area = float((cell_areas_da * (plot_dt[i] == 1)).sum())
        PR_area = float((cell_areas_da * (plot_dt[i]  == 2)).sum())
        PPETR_area = float((cell_areas_da * (plot_dt[i]  == 3)).sum())
        
        total_area = PET_area + PR_area + PPETR_area
        
        perc_PET = (PET_area / total_area) * 100
        perc_PR = (PR_area / total_area) * 100
        perc_PPETR = (PPETR_area / total_area) * 100
        
        values = [perc_PET, perc_PR, perc_PPETR]
        
        bar_data.append(values)
    
    x_values = np.arange(len(bar_data[0]))
    
    
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
            
            ax = axes[i]
           
            plot_dt[s].plot.imshow(ax=ax, cmap=cmap_highest_sim, extend='both', vmin=1, vmax=3)
            ax.add_feature(ocean)
            ax.coastlines()
            ax.set_title(f'{s}')
            
            # Create an inset for the bar plot
            left, bottom, width, height = 0.11, 0.16, 0.14, 0.22
            inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
            inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim, edgecolor='black')
            inset_ax.set_xlabel('')
            inset_ax.set_title('Area [%]')
            inset_ax.set_yticks([0, 25, 50, 75])
            inset_ax.set_xticks([])
            inset_ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Difference_Seasons.png', dpi=300)
    plt.show()
        
    
    
    ### analyse in addition if the Precipitation will increase or decrease:
    s_neg = season_modelmean_dpr < 0
    
    data_modified_ls = copy.deepcopy(high_sim_seasons)
    
    for season in seasons:
        print(f"Unique values in original {season}: {np.unique(high_sim_seasons[season])}")
        data_modified_ls[season] = high_sim_seasons[season].where(~s_neg.sel(season=season), high_sim_seasons[season] + 3)
        print(f"Unique values in modified {season}: {np.unique(data_modified_ls[season])}")
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = data_modified_ls.copy()
    plot_dt = {season: plot_dt[season].where(yearly_precip_threshold['pr']) for season in plot_dt}
    plot_dt = {
        season: plot_dt[season].where(seasons_change_greater['pr'].sel(season=season))
        for season in plot_dt}

    
    #Get barplot data:
    bar_data = []
    for i in seasons:
        PET_area_pos = float((cell_areas_da * (plot_dt[i] == 1)).sum())
        PR_area_pos = float((cell_areas_da * (plot_dt[i]  == 2)).sum())
        PPETR_area_pos = float((cell_areas_da * (plot_dt[i]  == 3)).sum())
        PET_area_neg = float((cell_areas_da * (plot_dt[i] == 4)).sum())
        PR_area_neg = float((cell_areas_da * (plot_dt[i]  == 5)).sum())
        PPETR_area_neg = float((cell_areas_da * (plot_dt[i]  == 6)).sum())
        
        total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg
        
        perc_PET_pos = (PET_area_pos / total_area) * 100
        perc_PR_pos = (PR_area_pos / total_area) * 100
        perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    
        perc_PET_neg = (PET_area_neg / total_area) * 100
        perc_PR_neg = (PR_area_neg / total_area) * 100
        perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    
        values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos,
                  perc_PET_neg, perc_PR_neg, perc_PPETR_neg]
        
        bar_data.append(values)
    
    x_values = np.arange(len(bar_data[0]))
    
    
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
            
            season_dt = plot_dt[s].squeeze()
            ax = axes[i]
           
            season_dt.plot.imshow(ax=ax, cmap=cmap_highest_sim_Pmask, extend='both', vmin=1, vmax=6)
            ax.add_feature(ocean)
            ax.coastlines()
            ax.set_title(f'{s}')
            
            #Inset for the bar plot
            left, bottom, width, height = 0.08, 0.16, 0.19, 0.22
            inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
            inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim_Pmask, edgecolor='black')
            inset_ax.set_xlabel('')
            inset_ax.set_title('Area [%]')
            inset_ax.set_yticks([0, 10, 20, 30, 40])
            inset_ax.set_xticks([])
            inset_ax.set_xticklabels([])
    
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Difference_Seasons_Pmask.png', dpi=300)
    plt.show()
    
    #################################
    #   High Similarity - Ratios    #
    #################################
    
    # calculate the ratio of changes:
    sdelta_ds['ratio_ETP'] = sdelta_ds['d_evspsbl']/sdelta_ds['d_pr']
    sdelta_ds['ratio_RP'] = sdelta_ds['d_mrro']/sdelta_ds['d_pr']     
    sdelta_ds['ratio_PETRP'] = sdelta_ds['d_PETR']/sdelta_ds['d_pr']
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = sdelta_ds.copy()
    plot_dt['ratio_ETP'] = plot_dt['ratio_ETP'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['ratio_RP'] = plot_dt['ratio_RP'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])
    plot_dt['ratio_PETRP'] = plot_dt['ratio_PETRP'].where(yearly_precip_threshold['pr'] & seasons_change_greater['pr'])

    fig, axes = plt.subplots(nrows=len(seasons), ncols=3, figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    titles_ratio = titles = [r'$\Delta ET / \Delta P $', r'$\Delta Q / \Delta P $', r'$\Delta RES / \Delta P $']
    for i, s in enumerate(seasons):
        for j, title, variable in zip([0,1,2,3], titles_ratio, ['ratio_ETP', 'ratio_RP', 'ratio_PETRP']):
            ax = axes[i, j]
            abs(plot_dt[variable]).sel(season=s).plot.imshow(ax=ax, cmap=pal_RdBu, extend='both',
                                                    #center=0,
                                                    #vmin=-5e-6,
                                                    vmin=-2,
                                                    vmax=+2,
                                                    cbar_kwargs={'label': f'{title}'})
            ax.coastlines()
            ax.set_title(f'{title} - {s}')
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_Ratio_PVars_Seasons.png', dpi=300, transparent=True)
    plt.show()
    
    high_sim_seasons = {}
    for s in seasons: 
        highest_sim = ratio1_index_allcells(data = sdelta_ds.where(sdelta_ds['season'] ==s, drop=True), var1='ratio_ETP', var2 = 'ratio_RP', var3= 'ratio_PETRP')
        high_sim_seasons[s] = highest_sim
        
    #get rid of season dimension:
    high_sim_seasons = {s: da.squeeze('season') for s, da in high_sim_seasons.items()}
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = high_sim_seasons.copy()
    plot_dt = {season: plot_dt[season].where(yearly_precip_threshold['pr']) for season in plot_dt}
    plot_dt = {
        season: plot_dt[season].where(seasons_change_greater['pr'].sel(season=season))
        for season in plot_dt}

    
    bar_data = []
    for i in seasons:
        PET_area = float((cell_areas_da * (plot_dt[i] == 1)).sum())
        PR_area = float((cell_areas_da * (plot_dt[i]  == 2)).sum())
        PPETR_area = float((cell_areas_da * (plot_dt[i]  == 3)).sum())
        
        total_area = PET_area + PR_area + PPETR_area 
        
        perc_PET = (PET_area / total_area) * 100
        perc_PR = (PR_area / total_area) * 100
        perc_PPETR = (PPETR_area / total_area) * 100
        
        values = [perc_PET, perc_PR, perc_PPETR]
        
        bar_data.append(values)
    
    x_values = np.arange(len(bar_data[0]))
    
       
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
            
            ax = axes[i]
            plot_dt[s].plot.imshow(ax=ax, cmap=cmap_highest_sim, extend='both', 
                                                  cbar_kwargs={'label': 'Highest Similarity'}, 
                                                  vmin=1, vmax=3)
            ax.add_feature(ocean)
            ax.coastlines()
            ax.set_title(f'{s}')
            
            # Create an inset for the bar plot
            left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
            inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
            inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim, edgecolor='black')
            inset_ax.set_xlabel('')
            inset_ax.set_title('No. of cells [%]')
            inset_ax.set_yticks([0, 25, 50, 75])
            inset_ax.set_xticks([])
            inset_ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Ratio_Seasons.png', dpi=300)
    plt.show()
        
    
    
    ### analyse in addition if the Precipitation will increase or decrease:
    s_neg = season_modelmean_dpr < 0
    
    data_modified_ls = copy.deepcopy(high_sim_seasons)
    
    for season in seasons:
        print(f"Unique values in original {season}: {np.unique(high_sim_seasons[season])}")
        data_modified_ls[season] = high_sim_seasons[season].where(~s_neg.sel(season=season), high_sim_seasons[season] + 3)
        print(f"Unique values in modified {season}: {np.unique(data_modified_ls[season])}")
    
    # For Plotting mask out areas with too little total precipitation and to little change in precipitation:
    plot_dt = data_modified_ls.copy()
    plot_dt = {season: plot_dt[season].where(yearly_precip_threshold['pr']) for season in plot_dt}
    plot_dt = {
        season: plot_dt[season].where(seasons_change_greater['pr'].sel(season=season))
        for season in plot_dt}

    
    #Get barplot data:
    bar_data = []
    for i in seasons:
        PET_area_pos = float((cell_areas_da * (plot_dt[i] == 1)).sum())
        PR_area_pos = float((cell_areas_da * (plot_dt[i]  == 2)).sum())
        PPETR_area_pos = float((cell_areas_da * (plot_dt[i]  == 3)).sum())
        PET_area_neg = float((cell_areas_da * (plot_dt[i] == 4)).sum())
        PR_area_neg = float((cell_areas_da * (plot_dt[i]  == 5)).sum())
        PPETR_area_neg = float((cell_areas_da * (plot_dt[i]  == 6)).sum())
        
        total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg
        
        perc_PET_pos = (PET_area_pos / total_area) * 100
        perc_PR_pos = (PR_area_pos / total_area) * 100
        perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    
        perc_PET_neg = (PET_area_neg / total_area) * 100
        perc_PR_neg = (PR_area_neg / total_area) * 100
        perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    
        values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos,
                  perc_PET_neg, perc_PR_neg, perc_PPETR_neg]
        
        bar_data.append(values)
    
    x_values = np.arange(len(bar_data[0]))
    
    
    
    fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    
    for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
            
            season_dt = plot_dt[s].squeeze()
            ax = axes[i]
           
            season_dt.plot.imshow(ax=ax, cmap=cmap_highest_sim_Pmask, extend='both', 
                                                  cbar_kwargs={'label': 'Highest Similarity'},
                                                  vmin=1, vmax=6)
            ax.add_feature(ocean)
            ax.coastlines()
            ax.set_title(f'{s}')
            
            # Create an inset for the bar plot
            left, bottom, width, height = 0.10, 0.12, 0.12, 0.24
            inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
            inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim_Pmask, edgecolor='black')
            inset_ax.set_xlabel('')
            inset_ax.set_title('No. of cells [%]')
            inset_ax.set_yticks([0, 10, 20, 30, 40])
            inset_ax.set_xticks([])
            inset_ax.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'{model_name}_HighSimilarity_Ratio_Seasons_Pmask.png', dpi=300)
    plt.show()
    
    
    # collect and save selected data:
    # Concatenate the DataArray objects along a new 'season' dimension
    sim_change = xr.concat([high_sim_seasons[season] for season in high_sim_seasons], dim='season')
    sim_change_Pmask = xr.concat([data_modified_ls[season] for season in data_modified_ls], dim='season')
    sdelta_ds = sdelta_ds.assign(sim_change=sim_change)
    sdelta_ds = sdelta_ds.assign(sim_change_Pmask=sim_change_Pmask)
    sdelta = sdelta_ds[['d_pr', 'd_mrro', 'd_evspsbl', 'd_PETR',
                      'ratio_ETP', 'ratio_RP', 'ratio_PETRP', 'sim_change', 'sim_change_Pmask']]
    sdelta.to_netcdf(data_path + model_name + '_sdelta.nc')
    
    #all_models_sdelta.append(sdelta)
    all_models_sdelta.append({'name': model_name, 'delta': sdelta})
    
    
    print(f'stored sdelta data for model {model_name}.')


# save all models data:

# Save the list to a file
with open(data_path + 'all_models_delta.pkl', 'wb') as f:
    pickle.dump(all_models_delta, f)

# Save the list to a file
with open(data_path + 'all_models_sdelta.pkl', 'wb') as f:
    pickle.dump(all_models_sdelta, f)


'''
################### Code to show 'too close' differences:
# Sum the absolute changes across the variables: When shared delta is lower than 5 %, then it is considered as clearly highest similarity:
threshhold = 0.05

delta_ds['sim_change'] = sim_change

abs_values = xr.Dataset({
    'abs_delta_PET': np.abs(delta_ds['delta_PET']),
    'abs_delta_PR': np.abs(delta_ds['delta_PR']),
    'abs_delta_PPETR': np.abs(delta_ds['delta_PPETR']),
})

delta_ds['sum_changes'] = abs_values.to_array(dim='variable').sum(dim='variable')
delta_ds[['delta_PET', 'delta_PR', 'delta_PPETR', 'sum_changes', 'sim_change']].sel(lat=-10, lon=-50)  
np.unique(delta_ds['sim_change'])



PET_change = delta_ds.where(delta_ds['sim_change'] == 1)
PET_change['sim_change'].plot.imshow()
ratio_PET = abs(PET_change['delta_PET'])/PET_change['sum_changes']
ratio_PET.plot.imshow()
mask_PET = ratio_PET.where(ratio_PET <= threshhold)
mask_PET = ~((ratio_PET) <= threshhold) # values that are >= 0.1 get True and all others get False
mask_PET.plot.imshow()

PR_change = delta_ds.where(delta_ds['sim_change'] == 2)
PR_change['sim_change'].plot.imshow()

ratio_PR = abs(PR_change['delta_PR'])/PR_change['sum_changes']
ratio_PR.plot.imshow()
mask_PR = ratio_PR.where(ratio_PR <= threshhold)
mask_PR = ~((ratio_PR) <= threshhold)
mask_PR.plot.imshow()

PPETR_change = delta_ds.where(delta_ds['sim_change'] == 3)
PPETR_change['sim_change'].plot.imshow()

ratio_PPETR = abs(PPETR_change['delta_PPETR'])/PPETR_change['sum_changes']
ratio_PPETR.plot.imshow()
mask_PPETR = ratio_PPETR.where(ratio_PPETR <= threshhold)
mask_PPETR = ~((ratio_PR) <= threshhold)
mask_PPETR.plot.imshow()

mask_combined = ~(mask_PET.astype(bool) & mask_PR.astype(bool) & mask_PPETR.astype(bool))
mask_combined.plot.imshow()

np.unique(mask_combined)


sim_change2 = sim_change.where(~mask_combined, 4)
np.unique(sim_change2)
delta_ds['clear_sim_change'] = sim_change2

############## Plot most similar changes:
# barplot data:
np.unique(sim_change2)
colors = ['forestgreen', 'skyblue', 'darkgoldenrod', 'black']    

PET_cells = np.sum(sim_change2 == 1)
PR_cells = np.sum(sim_change2 == 2)
PPETR_cells = np.sum(sim_change2 == 3)
not_clear = np.sum(sim_change2 == 4)

total_cells = PET_cells + PR_cells + PPETR_cells + not_clear
perc_PET = (PET_cells / total_cells) * 100
perc_PR = (PR_cells / total_cells) * 100
perc_PPETR = (PPETR_cells / total_cells) * 100
perc_notclear = (not_clear / total_cells) * 100

categories = ['delta_PET', 'delta_PR', 'delta_PPETR', 'not_clear']
values = [perc_PET, perc_PR, perc_PPETR, perc_notclear]

cmap = ListedColormap(colors)
cmap.set_bad('lightgray')
  

## Plot:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

sim_change2.plot.imshow(ax=axes, cmap=cmap, extend='both', 
                                      cbar_kwargs={'label': 'Highest Similarity'},
                                      vmin=1, vmax=4)
axes.coastlines()
axes.set_title('Yearly')

left, bottom, width, height = 0.13, 0.15, 0.10, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
ax_inset.bar(categories, values, color=colors, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('No. of cells [%]')
ax_inset.set_yticks([0, 20, 40, 60])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig('Plots/Global/High_Similarity_Yearly.png', dpi=300)
plt.show()

### analyse in addition if the Precipitation will increase or decrease:
m_neg = delta_ds['d_pr'] < 0

np.unique(sim_change2)
data_modified = sim_change.where(~m_neg, sim_change2+4)
np.unique(data_modified)

#Barplot global:
PET_cells_pos = np.sum(data_modified == 1)
PR_cells_pos = np.sum(data_modified == 2)
PPETR_cells_pos = np.sum(data_modified ==3)

PET_cells_neg = np.sum(data_modified == 5)
PR_cells_neg = np.sum(data_modified == 6)
PPETR_cells_neg = np.sum(data_modified ==7)

not_clear = np.sum(data_modified == 4|8)


total_cells = PET_cells_pos + PPETR_cells_pos + PR_cells_pos + PET_cells_neg + PPETR_cells_neg + PR_cells_neg + not_clear

perc_PET_pos = (PET_cells_pos / total_cells) * 100
perc_PPETR_pos = (PPETR_cells_pos / total_cells) * 100
perc_PR_pos = (PR_cells_pos / total_cells) * 100
perc_not_clear = (not_clear / total_cells) * 100

perc_PET_neg = (PET_cells_neg / total_cells) * 100
perc_PPETR_neg = (PPETR_cells_neg / total_cells) * 100
perc_PR_neg = (PR_cells_neg / total_cells) * 100


categories = ['$\Delta$ET-$\Delta$P pos', '$\Delta$R-$\Delta$P pos', '$\Delta$(P-ET-R)-$\Delta$P pos', 
              '$\Delta$ET-$\Delta$P neg', '$\Delta$R-$\Delta$P neg', '$\$\Delta$(P-ET-R)-$\Delta$P neg', 'not clear']
values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, 
          perc_PET_neg, perc_PR_neg, perc_PPETR_neg, not_clear]


colors = ['forestgreen', 'steelblue', 'darkgoldenrod', 'black', 'lightgreen', 'skyblue','wheat', 'black'] 

## Plot:
cmap = ListedColormap(colors)
cmap.set_bad('lightgray')

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

# Plot the main map
data_modified.plot.imshow(ax=axes, cmap=cmap,  extend='both', 
                                      cbar_kwargs={'label': 'Highest Change'}, 
                                      vmin=1, vmax=8)
axes.coastlines()
axes.set_title('Similar Change, Annual Mean')

# Create an inset for the bar plot
left, bottom, width, height = 0.13, 0.1, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')

# Plot the bar plot
ax_inset.bar(categories, values, color=colors, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('No. of cells [%]')

# Hide xticks and xtick labels
ax_inset.set_yticks([0, 10, 20, 30, 40])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig('Plots/Global/High_Similarity_Yearly_Pmask.png', dpi=300)
plt.show()


######## same for seasons #######################################################

for i, s in zip([0,1,2,3], seasons): 
        
    threshhold = 0.05
    
    season_dt = sdelta_ds.sel(season=s)    
    season_dt['sim_change'] = high_sim_seasons[s]
    
    abs_values = xr.Dataset({
        'abs_delta_PET': np.abs(season_dt['delta_PET']),
        'abs_delta_PR': np.abs(season_dt['delta_PR']),
        'abs_delta_PPETR': np.abs(season_dt['delta_PPETR']),
    })
    
    season_dt['sum_changes'] = abs_values.to_array(dim='variable').sum(dim='variable')
    season_dt[['delta_PET', 'delta_PR', 'delta_PPETR', 'sum_changes', 'sim_change']].sel(lat=-10, lon=-50)
    np.unique(season_dt['sim_change'])
    
    
    
    PET_change = season_dt.where(season_dt['sim_change'] == 1)
    PET_change['sim_change'].plot.imshow()
    ratio_PET = abs(PET_change['delta_PET'])/PET_change['sum_changes']
    ratio_PET.plot.imshow()
    mask_PET = ratio_PET.where(ratio_PET <= threshhold)
    mask_PET = ~((ratio_PET) <= threshhold) # values that are >= 0.1 get True and all others get False
    
    PR_change = season_dt.where(season_dt['sim_change'] == 2)
    PR_change['sim_change'].plot.imshow()
    
    ratio_PR = abs(PR_change['delta_PR'])/PR_change['sum_changes']
    ratio_PR.plot.imshow()
    mask_PR = ratio_PR.where(ratio_PR <= threshhold)
    mask_PR = ~((ratio_PR) <= threshhold)
    
    PPETR_change = season_dt.where(season_dt['sim_change'] == 3)
    PPETR_change['sim_change'].plot.imshow()
    
    ratio_PPETR = abs(PPETR_change['delta_PPETR'])/PPETR_change['sum_changes']
    ratio_PPETR.plot.imshow()
    mask_PPETR = ratio_PPETR.where(ratio_PPETR <= threshhold)
    mask_PPETR = ~((ratio_PR) <= threshhold)
    
    mask_combined = ~(mask_PET.astype(bool) & mask_PR.astype(bool) & mask_PPETR.astype(bool))
    mask_combined.plot.imshow()
    
    high_sim_seasons[s] = high_sim_seasons[s].where(~mask_combined, 4)
    print(np.unique(high_sim_seasons[s]))
    
    
bar_data = []
for i in seasons:
    PET_cells = np.sum(high_sim_seasons[i] == 1)
    PR_cells = np.sum(high_sim_seasons[i] == 2)
    PPETR_cells = np.sum(high_sim_seasons[i] == 3)
    not_clear = np.sum(high_sim_seasons[i] == 4)
    
    total_cells = PET_cells + PPETR_cells + PR_cells + not_clear
    
    perc_PET = (PET_cells / total_cells) * 100
    perc_PPETR = (PPETR_cells / total_cells) * 100
    perc_PR = (PR_cells / total_cells) * 100
    perc_not_clear = (not_clear / total_cells) * 100
    
    values = [perc_PET.item(), perc_PR.item(), perc_PPETR.item(), perc_not_clear.item()]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))


colors = ['forestgreen', 'skyblue', 'darkgoldenrod', 'black']        
cmap = ListedColormap(colors)
cmap.set_bad('lightgray')
    
fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]
       
        high_sim_seasons[s].plot.imshow(ax=ax, cmap=cmap, extend='both', 
                                              cbar_kwargs={'label': 'Highest Similarity'}, 
                                              vmin=1, vmax=4)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        
        # Plot the bar plot
        inset_ax.bar(x_values, bar_data[i], color=colors, edgecolor='black')
        
        inset_ax.set_xlabel('')
        inset_ax.set_title('No. of cells [%]')
        
        # Hide xticks and xtick labels
        inset_ax.set_yticks([0, 25, 50, 75])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])


plt.tight_layout()
plt.savefig('Plots/Global/High_Similarity_Seasons.png', dpi=300)
plt.show()


'''
