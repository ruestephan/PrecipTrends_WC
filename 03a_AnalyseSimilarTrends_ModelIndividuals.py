# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:49:43 2024

@author: Ruth Stephan

I need to add a check if files are already available..
"""


#packages:
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import statsmodels.api as sm
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import seaborn as sns
import copy




#paths:
os.getcwd()
data_path = '/mnt/share/scratch/rs1155/data/CMIP6/output_data/'
plot_path = '/mnt/share/scratch/rs1155/plots/model_individuals/'


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
colors_highest_sim2 = ['forestgreen', 'skyblue', 'darkgoldenrod']
cmap_highest_sim2 = ListedColormap(colors_highest_sim2)
cmap_highest_sim2.set_bad('lightgray')
colors_highest_sim = ['forestgreen', 'skyblue', 'darkgoldenrod', 'black']
cmap_highest_sim = ListedColormap(colors_highest_sim)
cmap_highest_sim.set_bad('lightgray')
colors_highest_sim_Pmask = ['forestgreen', 'steelblue', 'darkgoldenrod', 'black', 'lightgreen', 'skyblue','wheat', 'darkgray'] 
cmap_highest_sim_Pmask = ListedColormap(colors_highest_sim_Pmask)
cmap_highest_sim_Pmask.set_bad('lightgray')
colors_highest_sim_Pmask2 = ['forestgreen', 'steelblue', 'darkgoldenrod', 'lightgreen', 'skyblue','wheat'] 
cmap_highest_sim_Pmask2 = ListedColormap(colors_highest_sim_Pmask2)
cmap_highest_sim_Pmask2.set_bad('lightgray')


'''
Functions
'''
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


def trend_single_cell(var, time):
    
    # Remove NaN values from both arrays
    valid_mask = ~np.isnan(var)
    var_valid = var[valid_mask]
    time_valid = time[valid_mask]
    
    if var_valid.size == 0:
     # If there are no valid values, return an array of NaNs with the same shape as var
     return np.full_like(var, np.nan)
    
    # Compute the lowess 
    lowess_result1 = sm.nonparametric.lowess(var_valid, time_valid, frac=0.25)
    y_lowess = lowess_result1[:, 1]
    
    # Reconstruct the array with NaNs in the original places
    result = np.full_like(var, np.nan)
    result[valid_mask] = y_lowess
    
    return result

def trend_all_cells(data, var, start_time, end_time):
    
    #data= detrended_ds 
    #var='pr' 
    #start_time = '2000'
    #end_time = '2100'
    
    data = data.sel(time=slice(start_time, end_time))
    numeric_time = np.arange(data.time.size)

    input_vars = [data[var], numeric_time]
    
    detrend_results = xr.apply_ufunc(
        trend_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        output_core_dims=[['time']],
        vectorize=True,
        keep_attrs=True
        )
    
    return detrend_results

def trend_seasons_all_cells(data, var, start_time, end_time, season):
    
    #data= detrended_ds 
    #var='pr' 
    #start_time = '2000'
    #end_time = '2100'
    #season = 'DJF'
    
    data = data.sel(time=slice(start_time, end_time))
    seasonal_data = data.where(data['time'].dt.season == season, drop=True)
    numeric_time = np.arange(seasonal_data.time.size)

    input_vars = [seasonal_data[var], numeric_time]
    
    detrend_results = xr.apply_ufunc(
        trend_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        output_core_dims=[['time']],
        vectorize=True,
        keep_attrs=True
        )
    
    return detrend_results

# Get to the files that should be uses:
files_in_directory = os.listdir(data_path)
combined_files = [file for file in files_in_directory if 'combined' in file]
model_names = [filename.split('_combined.nc')[0] for filename in combined_files]    

# I need a check for which models trends are already calculated. And the remove them:
#models_to_remove = ['CAMS-CSM1-0', 'CESM2', 'CESM2-WACCM', 'FGOALS-g3']
#model_names = model_names[~np.isin(model_names, models_to_remove)]



all_models_delta = []
all_models_sdelta = []
    
model_name = model_names[0]
  
combined = xr.open_dataset(data_path + model_name + '_combined.nc')
combined = combined.sel(time=slice('2000', '2100'))

# calculate area for each gricell and store that in a 2nd array:
latitudes = combined['lat'].values
longitudes = combined['lon'].values
cell_areas = calculate_cell_area(latitudes, longitudes)
cell_areas_da = xr.DataArray(cell_areas, dims=['lat', 'lon'], coords={'lat': latitudes, 'lon': longitudes})


trended_ds = combined.copy()
trended_ds = trended_ds.fillna(np.nan)

DJF_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'DJF', drop=True)
MAM_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'MAM', drop=True)
JJA_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'JJA', drop=True)
SON_trended_ds = trended_ds.where(trended_ds['time'].dt.season == 'SON', drop=True)


trended_ds['tas'] = trend_all_cells(combined, 'tas', start_time='2000', end_time='2100')
trended_ds['pr'] = trend_all_cells(combined, 'pr', start_time='2000', end_time='2100')
trended_ds['evspsbl'] = trend_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100')
trended_ds['mrso'] = trend_all_cells(combined, 'mrso', start_time='2000', end_time='2100')
trended_ds['mrro'] = trend_all_cells(combined, 'mrro', start_time='2000', end_time='2100')
trended_ds['PETR'] = trend_all_cells(combined, 'PETR', start_time='2000', end_time='2100')

trend = trended_ds
trend.to_netcdf(data_path + model_name + '_trend.nc')

# Plot the original and smoothed data for a specific cell
combined['mrso'].sel(time=slice('2050', '2100')).mean(dim=['lat', 'lon']).plot(label='Original Data')
trend['mrso'].sel(time=slice('2050', '2100')).mean(dim=['lat', 'lon']).plot(label='Detrended Data')
plt.legend()
plt.show()

combined['tas'].sel(time=slice('2050', '2100')).mean(dim=['lat', 'lon']).plot(label='Original Data')
trend['tas'].sel(time=slice('2050', '2100')).mean(dim=['lat', 'lon']).plot(label='Detrended Data')
plt.legend()
plt.show()


DJF_trended_ds['tas'] = trend_seasons_all_cells(combined, 'tas', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['pr'] = trend_seasons_all_cells(combined, 'pr', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['evspsbl'] = trend_seasons_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['mrso'] = trend_seasons_all_cells(combined, 'mrso', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['mrro'] = trend_seasons_all_cells(combined, 'mrro', start_time='2000', end_time='2100', season='DJF')
DJF_trended_ds['PETR'] = trend_seasons_all_cells(combined, 'PETR', start_time='2000', end_time='2100', season='DJF')

MAM_trended_ds['tas'] = trend_seasons_all_cells(combined, 'tas', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['pr'] = trend_seasons_all_cells(combined, 'pr', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['evspsbl'] = trend_seasons_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['mrso'] = trend_seasons_all_cells(combined, 'mrso', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['mrro'] = trend_seasons_all_cells(combined, 'mrro', start_time='2000', end_time='2100', season='MAM')
MAM_trended_ds['PETR'] = trend_seasons_all_cells(combined, 'PETR', start_time='2000', end_time='2100', season='MAM')

JJA_trended_ds['tas'] = trend_seasons_all_cells(combined, 'tas', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['pr'] = trend_seasons_all_cells(combined, 'pr', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['evspsbl'] = trend_seasons_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['mrso'] = trend_seasons_all_cells(combined, 'mrso', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['mrro'] = trend_seasons_all_cells(combined, 'mrro', start_time='2000', end_time='2100', season='JJA')
JJA_trended_ds['PETR'] = trend_seasons_all_cells(combined, 'PETR', start_time='2000', end_time='2100', season='JJA')

SON_trended_ds['tas'] = trend_seasons_all_cells(combined, 'tas', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['pr'] = trend_seasons_all_cells(combined, 'pr', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['evspsbl'] = trend_seasons_all_cells(combined, 'evspsbl', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['mrso'] = trend_seasons_all_cells(combined, 'mrso', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['mrro'] = trend_seasons_all_cells(combined, 'mrro', start_time='2000', end_time='2100', season='SON')
SON_trended_ds['PETR'] = trend_seasons_all_cells(combined, 'PETR', start_time='2000', end_time='2100', season='SON')


def plot_longterm_trend():
    #lat = 48
    #lon = 2
    
    pr_dt = trend['pr'].mean(dim=['lat', 'lon'])
    evspsbl_dt = trend['evspsbl'].mean(dim=['lat', 'lon'])
    mrro_dt = trend['mrro'].mean(dim=['lat', 'lon'])
    PETR_dt = trend['PETR'].mean(dim=['lat', 'lon'])
    
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.figure(figsize=(12, 8))
    pr_dt.sel(time=slice('2000', '2100')).plot(label='Precipitation')
    evspsbl_dt.sel(time=slice('2000', '2100')).plot(label='Evapotranspiration')
    mrro_dt.sel(time=slice('2000', '2100')).plot(label='Runoff')
    PETR_dt.sel(time=slice('2000', '2100')).plot(label='Residual')
    plt.xlabel('Time')
    plt.title('Mean Trends')
    plt.legend()
    plt.savefig(plot_path + 'Trend_AllVars_Yearly.png', dpi=300)
    plt.show()

def plot_seasonal_longterm_trend(var):
    #var = 'tas'
    
    DJF_dt = DJF_trended_ds[var].mean(dim=['lat', 'lon'])
    MAM_dt = MAM_trended_ds[var].mean(dim=['lat', 'lon'])
    JJA_dt = JJA_trended_ds[var].mean(dim=['lat', 'lon'])
    SON_dt = SON_trended_ds[var].mean(dim=['lat', 'lon'])
    
    # Create numeric time arrays
    numeric_time_DJF = np.arange(DJF_dt.time.size)
    numeric_time_MAM = np.arange(MAM_dt.time.size)
    numeric_time_JJA = np.arange(JJA_dt.time.size)
    numeric_time_SON = np.arange(SON_dt.time.size)
    
    # Create the plot
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    plt.figure(figsize=(12, 8))
    
    # Plot each dataset
    plt.plot(numeric_time_DJF, DJF_dt, label='DJF')
    plt.plot(numeric_time_MAM, MAM_dt, label='MAM')
    plt.plot(numeric_time_JJA, JJA_dt, label='JJA')
    plt.plot(numeric_time_SON, SON_dt, label='SON')
    
    # Add labels and title
    plt.xlabel('Time')
    plt.ylabel(f'{var}')
    plt.title(f'{var} mean trend for different seasons')
    plt.legend()
    plt.savefig(plot_path + f'Trend_{var}_Seasons.png', dpi=300)
    plt.show()


plot_longterm_trend()

plot_seasonal_longterm_trend(var='tas')
plot_seasonal_longterm_trend(var='pr')
plot_seasonal_longterm_trend(var='mrro')
plot_seasonal_longterm_trend(var='PETR')

#save the data:
DJF_trend = DJF_trended_ds    
MAM_trend = MAM_trended_ds    
JJA_trend = JJA_trended_ds    
SON_trend = SON_trended_ds    
    
DJF_trend.to_netcdf(data_path + model_name + '_DJF_trend.nc')
MAM_trend.to_netcdf(data_path + model_name + '_MAM_trend.nc')
JJA_trend.to_netcdf(data_path + model_name + '_JJA_trend.nc')
SON_trend.to_netcdf(data_path + model_name + '_SON_trend.nc')




##################################################################
##    Calculate Correlations for those longterm-trends:         ##
##################################################################

# load the data:
trend = xr.open_dataset(data_path + model_name + '_trend.nc')
DJF_trend = xr.open_dataset(data_path + model_name + '_DJF_trend.nc')
MAM_trend = xr.open_dataset(data_path + model_name + '_MAM_trend.nc')
JJA_trend = xr.open_dataset(data_path + model_name + '_JJA_trend.nc')
SON_trend = xr.open_dataset(data_path + model_name + '_SON_trend.nc')


def corr_func_single_cell(var1: np.ndarray, var2: np.ndarray) -> np.float64:
    # Remove NaN values from both arrays
    valid_mask = ~np.isnan(var1) & ~np.isnan(var2)
    var1_valid = var1[valid_mask]
    var2_valid = var2[valid_mask]
    
    #if len(var1_valid) == 0 or len(var2_valid) == 0:
    #    return np.nan
    
    # Compute the correlation coefficient
    correlation_coefficient = np.corrcoef(var1_valid, var2_valid)[0, 1]
    return correlation_coefficient


def corr_all_cells(data, var1, var2
                   #, start_time, end_time
                   ):
    
    #data= detrended_ds 
    #var1='d_pr' 
    #var2='d_evspsbl'
    
    #data = data.sel(time=slice(start_time, end_time))

    input_vars = [data[var1], data[var2]]
    
    corr_results = xr.apply_ufunc(
        corr_func_single_cell,
        *input_vars,
        input_core_dims=[['time'], ['time']],
        output_dtypes=[np.float64],
        vectorize=True,
        )
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    corr_results.plot.imshow(ax=axes, cmap=pal_RdBu, extend='both',
                                  center=0, vmin=-1, vmax=1,
                                  cbar_kwargs={'label': 'Pearson Correlation', 'shrink' : 0.9})
    axes.add_feature(ocean)
    axes.coastlines()
    axes.set_title(f'Correlation between {var1} and {var2}')
    plt.tight_layout()
    plt.savefig(plot_path + f'Corr_{var1}_{var2}.png', dpi=300, transparent=True)
    plt.show()
    
    return corr_results


def find_high_index(arr1, arr2, arr3, threshold = 0.05):
    # Mask NaN values in all arrays
    mask_arr1 = np.isnan(arr1)
    mask_arr2 = np.isnan(arr2)
    mask_arr3 = np.isnan(arr3)
    
    # Create masked arrays to handle NaN values
    arr1_masked = np.ma.masked_array(arr1, mask_arr1)
    arr2_masked = np.ma.masked_array(arr2, mask_arr2)
    arr3_masked = np.ma.masked_array(arr3, mask_arr3)
    
    # Find the maximum value among non-NaN value
    max_arr = np.maximum(abs(arr1_masked), np.maximum(abs(arr2_masked), abs(arr3_masked)))
    
    # Determine which array has the maximum value
    index_arr = np.where(max_arr >= 0.5, 
                         np.where(max_arr == abs(arr1_masked), 1, 
                                  np.where(max_arr == abs(arr2_masked), 2, 3)), 
                         np.nan).astype(float)
    
    ''' Do that with the model agreement
    #Analyse whether the highest correlation too close to the 2nd highest correlation:
    # 1. get the second closest to 1:
    second_arr = np.where(abs(arr1_masked) == max_arr, np.maximum(abs(arr2_masked), abs(arr3_masked)),
                           np.where(abs(arr2_masked) == max_arr, np.maximum(abs(arr1_masked), abs(arr3_masked)), np.maximum(abs(arr1_masked), abs(arr2_masked))))
    # 2. get the difference between the 1st and 2nd closest and the calculate the ratio of the difference and the 1st closest
    ratio = np.abs(max_arr - abs(second_arr)) / np.abs(max_arr)
    
    # 3. When this ratio is lower than 50 %, meaning that the 2nd closest to 1 needs to at least deviate with 50 % from the first --> unclear result:
    unclear = ratio < threshold
    index_arr[unclear] = 4
    '''
    
    # Set NaN values back to NaN
    index_arr[mask_arr1] = np.nan
    index_arr[mask_arr2] = np.nan
    index_arr[mask_arr3] = np.nan
    
    return index_arr

def high_corr_allcells(data, var1, var2, var3, save_string):
    input_vars = [abs(data[var1]), abs(data[var2]), abs(data[var3])]
    
    highest_corr = xr.apply_ufunc(
        find_high_index,
        *input_vars,
        input_core_dims=[['lat', 'lon'], ['lat', 'lon'], ['lat', 'lon']],
        output_core_dims=[['lat', 'lon']],
        output_dtypes=[float]
    )
    
    # Prepare barplot inset:
    colors = ['forestgreen', 'skyblue', 'darkgoldenrod']    

    PET_area = float((cell_areas_da * (highest_corr == 1)).sum())
    PR_area = float((cell_areas_da * (highest_corr == 2)).sum())
    PPETR_area = float((cell_areas_da * (highest_corr == 3)).sum())
    unclear_area = float((cell_areas_da * (highest_corr == 4)).sum())

    total_area = PET_area + PR_area + PPETR_area + unclear_area

    perc_PET = (PET_area / total_area) * 100
    perc_SM = (PR_area / total_area) * 100
    perc_PR = (PPETR_area / total_area) * 100
    perc_unclear = (PPETR_area / total_area) * 100

    categories = ['Cor(P,ET)', 'Cor(P,R)', 'Cor(P,SM)', 'unclear']
    values = [perc_PET, perc_PR, perc_SM, perc_unclear]
        
    cmap = ListedColormap(colors)
    cmap.set_bad('lightgray')


    ## Plot:
    sns.set(style="ticks")
    plt.rcParams['font.family'] = 'DejaVu Sans'
        
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

    highest_corr.plot.imshow(ax=axes, cmap=cmap_highest_sim, extend='both', cbar_kwargs={'label': 'Highest Correlation'})
    axes.add_feature(ocean)
    axes.coastlines()

    # Create an inset for the bar plot
    left, bottom, width, height = 0.1, 0.15, 0.10, 0.19
    ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
    ax_inset.bar(categories, values, color=colors_highest_sim, edgecolor='black')
    ax_inset.set_xlabel('')
    ax_inset.set_title('No. of cells [%]')
    ax_inset.set_yticks([0, 20, 40, 60])
    ax_inset.set_xticks([])
    ax_inset.set_xticklabels([])
    
    plt.tight_layout()
    plt.savefig(plot_path + f'High_Corr_{save_string}.png', dpi=300)
    plt.show()
    
    return highest_corr



corr_PET = corr_all_cells(data=trend, var1='pr', var2='evspsbl')
corr_PR = corr_all_cells(data=trend, var1='pr', var2='mrro')
corr_PPETR = corr_all_cells(data=trend, var1='pr', var2='PETR')

# Create a new xarray.Dataset
corr_ds = xr.Dataset({
    'corr_PET': corr_PET,
    'corr_PR': corr_PR,
    'corr_PPETR': corr_PPETR 
})

highest_corr = high_corr_allcells(corr_ds, 'corr_PET', 'corr_PR', 'corr_PPETR', save_string='Yearly')
np.unique(highest_corr)
highest_corr.mean(dim=['lat', 'lon'])
corr_PET.mean(dim=['lat', 'lon'])
corr_PR.mean(dim=['lat', 'lon'])
corr_PPETR.mean(dim=['lat', 'lon'])

fig, axs = plt.subplots(3, 1, figsize=(8, 12))  # 3 rows, 1 column

Cor_PET = np.corrcoef(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['evspsbl'].mean(dim=['lat', 'lon']))[0, 1]
axs[0].scatter(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['evspsbl'].mean(dim=['lat', 'lon']), color='lightblue')
axs[0].set_title(f'Correlation: {Cor_PET:.2f}')
axs[0].set_xlabel('pr')
axs[0].set_ylabel('evspsbl')
axs[0].grid(True)
axs[0].legend()

Cor_PR = np.corrcoef(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['mrro'].mean(dim=['lat', 'lon']))[0, 1]
axs[1].scatter(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['mrro'].mean(dim=['lat', 'lon']), color='purple')
axs[1].set_title(f'Correlation: {Cor_PR:.2f}')
axs[1].set_xlabel('pr')
axs[1].set_ylabel('mrro')
axs[1].grid(True)
axs[1].legend()

Cor_PPETR = np.corrcoef(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['PETR'].mean(dim=['lat', 'lon']))[0, 1]
axs[2].scatter(trended_ds['pr'].mean(dim=['lat', 'lon']), trended_ds['PETR'].mean(dim=['lat', 'lon']), color='brown')
axs[2].set_title(f'Correlation: {Cor_PPETR:.2f}')
axs[2].set_xlabel('pr')
axs[2].set_ylabel('PETR')
axs[2].grid(True)
axs[2].legend()

plt.tight_layout()
plt.show()


### analyse in addition if the Precipitation will increase or decrease:
delta_ds = combined.copy()   


## Calculate Difference between 2100 and 2025
delta_ds = delta_ds.assign(d_pr = calc_vardiff(delta_ds, 'pr'))
#if change for precip is lower than 3% the cell is not relevant for the analysis:
pr_20002025 = delta_ds['pr'].sel(time=slice('2000', '2025')).mean(dim='time')
pr_20752100 = delta_ds['pr'].sel(time=slice('2075', '2100')).mean(dim='time')
change_greater = np.abs(delta_ds['d_pr'])/pr_20002025 >= 0.03
delta_ds['d_pr'] = delta_ds['d_pr'].where(change_greater)

    
m_neg = delta_ds['d_pr'] < 0

np.unique(highest_corr)
data_modified = highest_corr.where(~m_neg, highest_corr+4)
np.unique(data_modified)


#Barplot global:
PET_area_pos = float((cell_areas_da * (data_modified == 1)).sum())
PR_area_pos = float((cell_areas_da * (data_modified == 2)).sum())
PPETR_area_pos = float((cell_areas_da * (data_modified == 3)).sum())
unclear_area_pos = float((cell_areas_da * (data_modified == 4)).sum())

PET_area_neg = float((cell_areas_da * (data_modified == 5)).sum())
PR_area_neg = float((cell_areas_da * (data_modified == 6)).sum())
PPETR_area_neg = float((cell_areas_da * (data_modified == 7)).sum())
unclear_area_neg = float((cell_areas_da * (data_modified == 8)).sum())

# Total area
total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + unclear_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg + unclear_area_neg

perc_PET_pos = (PET_area_pos / total_area) * 100
perc_PR_pos = (PR_area_pos / total_area) * 100
perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
perc_unclear_pos = (unclear_area_pos / total_area) * 100

perc_PET_neg = (PET_area_neg / total_area) * 100
perc_PR_neg = (PR_area_neg / total_area) * 100
perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
perc_unclear_neg = (unclear_area_neg / total_area) * 100


categories = ['Cor(P,ET) pos', 'Cor(P,R) pos', 'Cor(P,SM) pos', 'unclear pos',
              'Cor(P,ET) neg', 'Cor(P,R) neg', 'Cor(P,SM) neg', 'unclear neg']

values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, perc_unclear_pos,
          perc_PET_neg, perc_PR_neg, perc_PPETR_neg, perc_unclear_neg]


## Plot:
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 3), subplot_kw={'projection': ccrs.PlateCarree()})        

data_modified.plot.imshow(ax=axes, cmap=cmap_highest_sim_Pmask,  extend='both', 
                                      cbar_kwargs={'label': 'Highest Similarity'}, 
                                      vmin=1, vmax=8)
axes.add_feature(ocean)
axes.coastlines()
axes.set_title('Highest Similarity - Yearly')

# Create an inset for the bar plot
left, bottom, width, height = 0.11, 0.16, 0.14, 0.19
ax_inset = fig.add_axes([left, bottom, width, height], facecolor='white')
ax_inset.bar(categories, values, color=colors_highest_sim_Pmask, edgecolor='black')
ax_inset.set_xlabel('')
ax_inset.set_title('Area [%]')
ax_inset.set_yticks([0, 10, 20, 30, 40])
ax_inset.set_xticks([])
ax_inset.set_xticklabels([])

plt.tight_layout()
plt.savefig(plot_path + 'High_Correlation_Pmask_Yearly.png', dpi=300)
plt.show()





###################### seasons:

DJF_corr_PET = corr_all_cells(data=DJF_trend, var1='pr', var2='evspsbl')
DJF_corr_PR = corr_all_cells(data=DJF_trend, var1='pr', var2='mrro')
DJF_corr_PPETR = corr_all_cells(data=DJF_trend, var1='pr', var2='PETR')

DJF_corr_ds = xr.Dataset({
    'DJF_corr_PET': DJF_corr_PET,
    'DJF_corr_PR': DJF_corr_PR,
    'DJF_corr_PPETR': DJF_corr_PPETR 
})

DJF_highest_corr = high_corr_allcells(DJF_corr_ds, 'DJF_corr_PET', 'DJF_corr_PR', 'DJF_corr_PPETR', save_string='yearly')


MAM_corr_PET = corr_all_cells(data=MAM_trend, var1='pr', var2='evspsbl')
MAM_corr_PR = corr_all_cells(data=MAM_trend, var1='pr', var2='mrro')
MAM_corr_PPETR = corr_all_cells(data=MAM_trend, var1='pr', var2='PETR')

MAM_corr_ds = xr.Dataset({
    'MAM_corr_PET': MAM_corr_PET,
    'MAM_corr_PR': MAM_corr_PR,
    'MAM_corr_PPETR': MAM_corr_PPETR 
})

MAM_highest_corr = high_corr_allcells(MAM_corr_ds, 'MAM_corr_PET', 'MAM_corr_PR', 'MAM_corr_PPETR', save_string='yearly')

JJA_corr_PET = corr_all_cells(data=JJA_trend, var1='pr', var2='evspsbl')
JJA_corr_PR = corr_all_cells(data=JJA_trend, var1='pr', var2='mrro')
JJA_corr_PPETR = corr_all_cells(data=JJA_trend, var1='pr', var2='PETR')

JJA_corr_ds = xr.Dataset({
    'JJA_corr_PET': JJA_corr_PET,
    'JJA_corr_PR': JJA_corr_PR,
    'JJA_corr_PPETR': JJA_corr_PPETR 
})

JJA_highest_corr = high_corr_allcells(JJA_corr_ds, 'JJA_corr_PET', 'JJA_corr_PR', 'JJA_corr_PPETR', save_string='yearly')

SON_corr_PET = corr_all_cells(data=SON_trend, var1='pr', var2='evspsbl')
SON_corr_PR = corr_all_cells(data=SON_trend, var1='pr', var2='mrro')
SON_corr_PPETR = corr_all_cells(data=SON_trend, var1='pr', var2='PETR')

SON_corr_ds = xr.Dataset({
    'SON_corr_PET': SON_corr_PET,
    'SON_corr_PR': SON_corr_PR,
    'SON_corr_PPETR': SON_corr_PPETR 
})

SON_highest_corr = high_corr_allcells(SON_corr_ds, 'SON_corr_PET', 'SON_corr_PR', 'SON_corr_PPETR', save_string='yearly')

high_cor_seasons = xr.Dataset({
    'DJF' : DJF_highest_corr,
    'MAM' : MAM_highest_corr,
    'JJA' : JJA_highest_corr,
    'SON' : SON_highest_corr})

seasons = ['DJF', 'MAM', 'JJA', 'SON']

bar_data = []
for i in seasons:
    PET_area = float((cell_areas_da * (high_cor_seasons[i] == 1)).sum())
    PR_area = float((cell_areas_da * (high_cor_seasons[i]  == 2)).sum())
    PPETR_area = float((cell_areas_da * (high_cor_seasons[i]  == 3)).sum())
    unclear_area = float((cell_areas_da * (high_cor_seasons[i]  == 4)).sum())
    
    total_area = PET_area + PR_area + PPETR_area + unclear_area
    
    perc_PET = (PET_area / total_area) * 100
    perc_PR = (PR_area / total_area) * 100
    perc_PPETR = (PPETR_area / total_area) * 100
    perc_unclear = (unclear_area / total_area) * 100
    
    values = [perc_PET, perc_PR, perc_PPETR, perc_unclear]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))


    
fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
        ax = axes[i]
       
        high_cor_seasons[s].plot.imshow(ax=ax, cmap=cmap_highest_sim, extend='both', 
                                              cbar_kwargs={'label': 'Highest Correlation'}, 
                                              vmin=1, vmax=4)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.22
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim, edgecolor='black')
        inset_ax.set_xlabel('')
        inset_ax.set_title('Area [%]')
        inset_ax.set_yticks([0, 25, 50, 75])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])


plt.tight_layout()
plt.savefig(plot_path + 'High_Correlation_Seasons.png', dpi=300)
plt.show()


### analyse in addition if the Precipitation will increase or decrease:
sdelta_ds = combined.copy() 


## Calculate Difference between 2100 and 2025
sdelta_ds = sdelta_ds.assign(d_pr = calc_vardiff_season(sdelta_ds, 'pr'))
#if change for precip is lower than 5% the cell is not relevant for the analysis:
pr_20002025 = delta_ds['pr'].sel(time=slice('2000', '2025')).groupby('time.season').mean(dim='time')
pr_20752100 = delta_ds['pr'].sel(time=slice('2075', '2100')).groupby('time.season').mean(dim='time')
s_change_greater = np.abs(sdelta_ds['d_pr'])/pr_20002025 >= 0.03
sdelta_ds['d_pr'] = sdelta_ds['d_pr'].where(s_change_greater)

    
s_neg = sdelta_ds['d_pr'] < 0

data_modified_ls = copy.deepcopy(high_cor_seasons)

for season in seasons:
    print(f"Unique values in original {season}: {np.unique(high_cor_seasons[season])}")
    data_modified_ls[season] = high_cor_seasons[season].where(~s_neg.sel(season=season), high_cor_seasons[season] + 4)
    print(f"Unique values in modified {season}: {np.unique(data_modified_ls[season])}")


#Get barplot data:
bar_data = []
for i in seasons:
    PET_area_pos = float((cell_areas_da * (data_modified_ls[i] == 1)).sum())
    PR_area_pos = float((cell_areas_da * (data_modified_ls[i]  == 2)).sum())
    PPETR_area_pos = float((cell_areas_da * (data_modified_ls[i]  == 3)).sum())
    unclear_area_pos = float((cell_areas_da * (data_modified_ls[i]  == 4)).sum())

    PET_area_neg = float((cell_areas_da * (data_modified_ls[i] == 5)).sum())
    PR_area_neg = float((cell_areas_da * (data_modified_ls[i]  == 6)).sum())
    PPETR_area_neg = float((cell_areas_da * (data_modified_ls[i]  == 7)).sum())
    unclear_area_neg = float((cell_areas_da * (data_modified_ls[i]  == 8)).sum())
    
    total_area = PET_area_pos + PR_area_pos + PPETR_area_pos + unclear_area_pos + PET_area_neg + PR_area_neg + PPETR_area_neg + unclear_area_neg
    
    perc_PET_pos = (PET_area_pos / total_area) * 100
    perc_PR_pos = (PR_area_pos / total_area) * 100
    perc_PPETR_pos = (PPETR_area_pos / total_area) * 100
    perc_unclear_pos = (unclear_area_pos / total_area) * 100

    perc_PET_neg = (PET_area_neg / total_area) * 100
    perc_PR_neg = (PR_area_neg / total_area) * 100
    perc_PPETR_neg = (PPETR_area_neg / total_area) * 100
    perc_unclear_neg = (unclear_area_neg / total_area) * 100

    values = [perc_PET_pos, perc_PR_pos, perc_PPETR_pos, perc_unclear_pos,
              perc_PET_neg, perc_PR_neg, perc_PPETR_neg, perc_unclear_neg]
    
    bar_data.append(values)

x_values = np.arange(len(bar_data[0]))



fig, axes = plt.subplots(nrows=len(seasons), ncols=1, figsize=(7, 12), subplot_kw={'projection': ccrs.PlateCarree()})

for i, s  in zip([0,1,2,3], ['DJF', 'MAM', 'JJA', 'SON']):
        
    
        ax = axes[i]
        high_cor_seasons[s].plot.imshow(ax=ax, cmap=cmap_highest_sim_Pmask, extend='both', 
                                              cbar_kwargs={'label': 'Highest Similarity'},
                                              vmin=1, vmax=8)
        ax.add_feature(ocean)
        ax.coastlines()
        ax.set_title(f'{s}')
        
        # Create an inset for the bar plot
        left, bottom, width, height = 0.10, 0.12, 0.12, 0.24
        inset_ax = ax.inset_axes([left, bottom, width, height], facecolor='white')
        inset_ax.bar(x_values, bar_data[i], color=colors_highest_sim_Pmask, edgecolor='black')
        inset_ax.set_xlabel('')
        inset_ax.set_title('Area [%]')
        inset_ax.set_yticks([0, 10, 20, 30, 40])
        inset_ax.set_xticks([])
        inset_ax.set_xticklabels([])


plt.tight_layout()
plt.savefig(plot_path + 'High_Correlation_Pmask_Seasons.png', dpi=300)
plt.show()









''' old version with loop

def corr_func(data, var1, var2, time_start, time_end, save_string):
    import cartopy.crs as ccrs
    import xarray as xr
    
    #data = trended_ds
    #var1 = 'pr'
    #var2 = 'evspsbl'
    #time_start = '2000'
    #time_end = '2100'
    #save_string = 'PET_yearly'
    
    corr_results = xr.Dataset(
        {'corr': (['lat', 'lon'], np.random.rand(len(corr_ds.lat), len(corr_ds.lon)))},
        coords={'lat': corr_ds.lat, 'lon': corr_ds.lon}
    )
    
    for i, lat_val in enumerate(tqdm(corr_ds.lat)):
        for j, lon_val in enumerate(corr_ds.lon):
            # Select data for the current cell
            cell_data = data.sel(time=slice(time_start, time_end)).isel(lat=i, lon=j).to_dataframe()
            
            if ~np.all(np.isnan(cell_data[var1])):
                corr = cell_data[var1].corr(cell_data[var2])
            else:
                corr = np.nan
            # Store the lowess result in the array
            corr_results['corr'].loc[lat_val, lon_val] = corr
            
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 4),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    corr_results['corr'].plot.imshow(ax=axes, cmap='RdBu', extend='both',
                                  center=0, vmin=-1, vmax=1,
                                  cbar_kwargs={'label': 'Pearson Correlation'})
    axes.coastlines()
    axes.set_title(f'Correlation between {var1} and {var2}')
    plt.tight_layout()
    #plt.savefig(f'Plots/Global/{save_string}.png', dpi=300, transparent=True)
    plt.show()
            
    return corr_results

Cor_PET_loop = corr_func(data = trended_ds, var1='pr', var2 = 'evspsbl', time_start='2000', time_end='2100', save_string='Cor_PET_yearly')
Cor_PR_loop = corr_func(data = trended_ds, var1='pr', var2 = 'mrro', time_start='2000', time_end='2100', save_string='Cor_PR_yearly')
Cor_PPETR_loop = corr_func(data = trended_ds, var1='pr', var2 = 'PETR', time_start='2000', time_end='2100', save_string='Cor_PPETR_yearly')

# Extract the 'corr' DataArray from each Dataset
Cor_PR_da = Cor_PR_loop['corr']
Cor_PPETR_da = Cor_PPETR_loop['corr']
Cor_PET_da = Cor_PET_loop['corr']  # Assuming Cor_PET_loop is similarly defined

# Create a new Dataset combining the extracted DataArrays
corr_ds_loop = xr.Dataset({
    'Cor_PET_loop': Cor_PET_da,
    'Cor_PR_loop': Cor_PR_da,
    'Cor_PPETR_loop': Cor_PPETR_da
})



highest_corr_loop = high_corr_allcells(corr_ds_loop, 'Cor_PET_loop', 'Cor_PR_loop', 'Cor_PPETR_loop', save_string='yearly_loop')
np.unique(highest_corr)
highest_corr_loop.sel(lat=48, lon=2)
Cor_PET_loop.sel(lat=48, lon=2)
Cor_PR_loop.sel(lat=48, lon=2)
Cor_PPETR_loop.sel(lat=48, lon=2)

Cor_PET_loop.sel(lat=48, lon=2)
'''